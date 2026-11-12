"""PathPack-U, part 1: edit probabilities at three granularities.

The chunk model (history replay) gives P(chunk is edited). Two more models, trained on the same
replay tasks, give P(function is edited) and P(file is edited) from aggregates of the chunk
probabilities and chunk features. They are trained on out-of-fold chunk probabilities, grouped by
replay task, so no unit is scored by a model that saw its own task. No benchmark label is used.

The utility weights are not tuned: each is the reciprocal of the mean number of edited units per
replay task, so every term of the packing objective is an expected recall.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

from . import edit_model
from .edit_features import STAGE1

_COL = {name: i for i, name in enumerate(STAGE1)}
AGG = ('p_max', 'p_mean', 'p_sum', 'p_second', 'p_noisy_or', 'count', 'log_count',
       'lexical_max', 'semantic_max', 'symbol_max', 'graph_max', 'channels_max', 'log_tokens_max',
       'file_lexical_pct', 'file_semantic_pct')
PARAMS = dict(max_depth=3, learning_rate=0.05, max_iter=150, min_samples_leaf=40, l2_regularization=1.0,
              random_state=0)


def aggregate(p, X, groups):
    """One row of ``AGG`` features per group id ``0..groups.max()``."""
    p = np.asarray(p, dtype=np.float64)
    n_groups = int(groups.max()) + 1 if len(groups) else 0
    out = np.zeros((n_groups, len(AGG)))
    order = np.argsort(groups, kind='stable')
    bounds = np.searchsorted(groups[order], np.arange(n_groups + 1))
    for g in range(n_groups):
        rows = order[bounds[g]:bounds[g + 1]]
        if len(rows) == 0:
            continue
        pv = np.sort(p[rows])[::-1]
        x = X[rows]
        out[g] = [
            pv[0], pv.mean(), pv.sum(), pv[1] if len(pv) > 1 else 0.0, 1.0 - float(np.prod(1.0 - pv)),
            len(rows), np.log1p(len(rows)),
            x[:, _COL['lexical']].max(), x[:, _COL['semantic']].max(), x[:, _COL['symbol']].max(),
            x[:, _COL['graph']].max(), x[:, _COL['n_channels']].max(), x[:, _COL['log_function_tokens']].max(),
            x[:, _COL['file_lexical_pct']].min(), x[:, _COL['file_semantic_pct']].min(),
        ]
    return out


def _group_labels(y, groups):
    n_groups = int(groups.max()) + 1
    out = np.zeros(n_groups, dtype=int)
    np.maximum.at(out, groups, np.asarray(y, dtype=int))
    return out


class HierModel:
    def __init__(self, fn_model, file_model, weights, meta):
        self.fn_model, self.file_model, self.weights, self.meta = fn_model, file_model, weights, meta

    def predict(self, p, X, struct):
        """Return ``(p_function[n_fn], p_file[n_file])`` aligned with ``struct['fn']`` / ``struct['file']`` ids."""
        if len(p) == 0:
            return np.zeros(0), np.zeros(0)
        fn = self.fn_model.predict_proba(aggregate(p, X, struct['fn']))[:, 1]
        fl = self.file_model.predict_proba(aggregate(p, X, struct['file']))[:, 1]
        return fn, fl


def fit(tasks, *, folds=5, use_memory=False, meta=None):
    """``tasks`` are replay tasks (``X``, ``y``, ``struct``). The chunk model is refit per fold."""
    tasks = [t for t in tasks if len(t['y']) and t['y'].sum() > 0]
    groups = np.arange(len(tasks))
    fn_X, fn_y, file_X, file_y = [], [], [], []
    counts = {'chunk': [], 'fn': [], 'file': []}
    for train, test in GroupKFold(n_splits=folds).split(groups, groups, groups):
        chunk = edit_model.fit([tasks[i] for i in train], folds=3)
        for i in test:
            t = tasks[i]
            p = chunk.predict(t['X'], t['struct'], use_memory=use_memory)
            fn_X.append(aggregate(p, t['X'], t['struct']['fn']))
            fn_y.append(_group_labels(t['y'], t['struct']['fn']))
            file_X.append(aggregate(p, t['X'], t['struct']['file']))
            file_y.append(_group_labels(t['y'], t['struct']['file']))
            counts['chunk'].append(int(t['y'].sum()))
            counts['fn'].append(int(fn_y[-1].sum()))
            counts['file'].append(int(file_y[-1].sum()))
    fn_model = HistGradientBoostingClassifier(**PARAMS).fit(np.vstack(fn_X), np.concatenate(fn_y))
    file_model = HistGradientBoostingClassifier(**PARAMS).fit(np.vstack(file_X), np.concatenate(file_y))
    weights = {name: 1.0 / float(np.mean(values)) for name, values in counts.items()}
    info = {'tasks': len(tasks), 'folds': folds, 'use_memory': use_memory, 'params': PARAMS, 'agg': list(AGG),
            'mean_edited_units_per_task': {k: float(np.mean(v)) for k, v in counts.items()},
            'fn_rows': int(sum(len(v) for v in fn_y)), 'file_rows': int(sum(len(v) for v in file_y)), **(meta or {})}
    return HierModel(fn_model, file_model, weights, info)


def save(model, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('wb') as handle:
        pickle.dump(model, handle)


def load(path):
    """Load a model written by ``save`` on this machine. Pickle runs code on load: local files only."""
    path = Path(path)
    if not path.is_file():
        return None
    with path.open('rb') as handle:
        return pickle.load(handle)


def main(argv=None):
    import argparse
    import json
    from ..benchmark import read_jsonl
    parser = argparse.ArgumentParser(prog='pathpack-u-train')
    parser.add_argument('--train-dir', default='output/pathpack_e/train_history')
    parser.add_argument('--out', default='output/pathpack_e/model_hier.pkl')
    parser.add_argument('--exclude', nargs='+', required=True)
    args = parser.parse_args(argv)
    banned = {row['instance_id'] for path in args.exclude for row in read_jsonl(path)}
    tasks = []
    for path in sorted(Path(args.train_dir).glob('*.pkl')):
        with path.open('rb') as handle:  # local file written by history_replay
            tasks.append(pickle.load(handle))
    overlap = sorted({t['id'] for t in tasks} & banned)
    if overlap or any(t.get('supervision') != 'history_replay' for t in tasks):
        raise SystemExit(f'PathPack-U trains on history replay only; offending tasks: {overlap[:5]}')
    # Build through the importable module so the pickle does not point at ``__main__``.
    from codegraph_rag.pathpack_r import hier_model as module
    model = module.fit(tasks, meta={'train_dir': args.train_dir, 'supervision': 'history_replay'})
    module.save(model, args.out)
    print(json.dumps({'weights': model.weights, **{k: model.meta[k] for k in ('tasks', 'fn_rows', 'file_rows')}}))


if __name__ == '__main__':
    main()

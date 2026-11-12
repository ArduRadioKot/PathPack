"""Two-stage edit-location model for PathPack-E.

Stage 1: gradient boosting on candidate, file, function and git-history features.
Stage 2: the same learner on stage-1 features plus one round of call-edge message passing.
Stage-2 training uses out-of-fold stage-1 probabilities grouped by task, so it never sees a
stage-1 score that was fitted on its own task. Training data are train-pool tasks only.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

from .edit_features import HISTORY_COLUMNS, MEMORY_COLUMNS, STAGE1, STAGE2, stage2_features

PARAMS = dict(
    max_depth=4, learning_rate=0.05, max_iter=200, min_samples_leaf=40,
    l2_regularization=1.0, random_state=0,
)


def _learner():
    return HistGradientBoostingClassifier(**PARAMS)


class EditModel:
    def __init__(self, stage1, stage2, meta):
        self.stage1 = stage1
        self.stage2 = stage2
        self.meta = meta

    def predict(self, X, struct, *, use_graph=True, use_history=True, use_memory=True):
        X = np.array(X, dtype=np.float64, copy=True)
        if not use_history:
            X[:, list(HISTORY_COLUMNS)] = 0.0
        if not use_memory:
            X[:, list(MEMORY_COLUMNS)] = 0.0
        if len(X) == 0:
            return np.zeros(0)
        p0 = self.stage1.predict_proba(X)[:, 1]
        if not use_graph or self.stage2 is None:
            return p0
        X2 = np.hstack([X, stage2_features(p0, struct)])
        return self.stage2.predict_proba(X2)[:, 1]


def fit(tasks, *, folds=5, meta=None):
    """``tasks`` is a list of dicts: ``id``, ``X`` (n x f), ``y`` (n,), ``struct``."""
    tasks = [t for t in tasks if len(t['y']) and t['y'].sum() > 0]
    if len(tasks) < folds:
        raise ValueError(f'Need at least {folds} tasks with positives, got {len(tasks)}')
    X = np.vstack([t['X'] for t in tasks])
    y = np.concatenate([t['y'] for t in tasks]).astype(int)
    groups = np.concatenate([[i] * len(t['y']) for i, t in enumerate(tasks)])
    stage1 = _learner().fit(X, y)
    # Out-of-fold stage-1 probabilities for stage-2 training.
    oof = np.zeros(len(y))
    for train, test in GroupKFold(n_splits=folds).split(X, y, groups):
        oof[test] = _learner().fit(X[train], y[train]).predict_proba(X[test])[:, 1]
    parts, start = [], 0
    for t in tasks:
        n = len(t['y'])
        parts.append(stage2_features(oof[start:start + n], t['struct']))
        start += n
    stage2 = _learner().fit(np.hstack([X, np.vstack(parts)]), y)
    info = {
        'tasks': len(tasks), 'rows': int(len(y)), 'positives': int(y.sum()),
        'task_ids': [t['id'] for t in tasks], 'folds': folds, 'params': PARAMS,
        'stage1_features': list(STAGE1), 'stage2_features': list(STAGE2),
        **(meta or {}),
    }
    return EditModel(stage1, stage2, info)


def save(model, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('wb') as handle:
        pickle.dump(model, handle)


def load(path):
    """Load a model written by ``save`` on this machine.

    Pickle executes code on load. This is acceptable only because the file is produced locally by
    ``edit_train``; never point it at a model from another source.
    """
    path = Path(path)
    if not path.is_file():
        return None
    with path.open('rb') as handle:
        return pickle.load(handle)

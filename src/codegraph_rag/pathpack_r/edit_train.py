"""Build the PathPack-E training set and fit the edit-location model. Offline only.

The gold patch is read in exactly one place, ``label_regions``, and only for tasks listed in the
train pool. ``fit`` refuses any task that appears in a validation, holdout, diagnostic or earlier
test manifest. Inference never imports this module.
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np

from ..adaptive_chunker import build_chunk_index
from ..benchmark import read_jsonl
from ..config import V4Config
from ..experiments import load_scoring_dataset, verify_snapshot
from ..parser import scan_repository
from . import edit_model
from .config import PathPackRConfig
from .edit_features import build_stage1
from .history import load_history
from .retrieve import RepoIndex, build_pool
from .semantic import prepare_semantic
from .train import changed_base_lines

CHANNELS = ('lexical', 'symbol', 'semantic', 'structural')


def label_regions(regions, patch):
    """1 when the chunk's line range contains a removed line or an insertion anchor. Training only."""
    changed = changed_base_lines(patch)
    labels = np.zeros(len(regions), dtype=np.int8)
    for i, region in enumerate(regions):
        lines = changed.get(region.chunk.path, ())
        if any(region.chunk.line_start <= line <= region.chunk.line_end for line in lines):
            labels[i] = 1
    return labels


def build_task(row, patch, snapshots, config, v4, *, cache_dir, history_dir):
    repo = verify_snapshot(row, snapshots)
    graph = scan_repository(repo)
    chunks = build_chunk_index(repo, graph, v4)
    index = RepoIndex(chunks, graph, repo)
    index.semantic = prepare_semantic(
        index, row['problem_statement'], config, cache_dir=cache_dir, cache_key=row['instance_id'])
    if index.semantic is None or not index.semantic.available:
        raise RuntimeError(f'semantic channel unavailable for {row["instance_id"]}')
    regions, _report, _queries, _lexical = build_pool(index, row['problem_statement'], config, CHANNELS)
    history = load_history(repo, cache_dir=history_dir, cache_key=row['instance_id'])
    X, struct = build_stage1(regions, index, history, query=row['problem_statement'])
    y = label_regions(regions, patch)
    return {'id': row['instance_id'], 'repo': row['repo'], 'X': X, 'y': y, 'struct': struct,
            'history': history.source, 'regions': len(regions), 'positives': int(y.sum())}


def cmd_build(args):
    config = PathPackRConfig.load(args.config)
    v4 = V4Config.load(args.v4_config) if Path(args.v4_config).is_file() else V4Config()
    rows = read_jsonl(args.issues)
    patches = {r['instance_id']: r['patch'] for r in load_scoring_dataset()}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for number, row in enumerate(rows, 1):
        dest = out / f'{row["instance_id"]}.pkl'
        if dest.is_file():
            continue
        if args.limit and len(list(out.glob('*.pkl'))) >= args.limit:
            break
        started = time.perf_counter()
        try:
            task = build_task(row, patches[row['instance_id']], args.snapshots, config, v4,
                              cache_dir=args.cache, history_dir=args.history)
        except Exception as exc:  # keep going; a failed task is logged, never silently dropped
            print(f'[{number}/{len(rows)}] {row["instance_id"]} FAILED {type(exc).__name__}: {exc}', flush=True)
            continue
        with dest.open('wb') as handle:
            pickle.dump(task, handle)
        print(f'[{number}/{len(rows)}] {row["instance_id"]} regions={task["regions"]} '
              f'pos={task["positives"]} {time.perf_counter() - started:.0f}s', flush=True)


def cmd_warm(args):
    """Encode the semantic index and git history only. No label, no patch, no metric."""
    config = PathPackRConfig.load(args.config)
    v4 = V4Config.load(args.v4_config) if Path(args.v4_config).is_file() else V4Config()
    rows = read_jsonl(args.issues)
    for number, row in enumerate(rows, 1):
        started = time.perf_counter()
        repo = verify_snapshot(row, args.snapshots)
        graph = scan_repository(repo)
        index = RepoIndex(build_chunk_index(repo, graph, v4), graph, repo)
        prepare_semantic(index, row['problem_statement'], config, cache_dir=args.cache, cache_key=row['instance_id'])
        load_history(repo, cache_dir=args.history, cache_key=row['instance_id'])
        print(f'[{number}/{len(rows)}] warm {row["instance_id"]} {time.perf_counter() - started:.0f}s', flush=True)


def cmd_fit(args):
    banned = set()
    for path in args.exclude:
        for row in read_jsonl(path):
            banned.add(row['instance_id'])
    tasks = []
    dirs = args.train_dir if isinstance(args.train_dir, (list, tuple)) else [args.train_dir]
    for folder in dirs:
        for path in sorted(Path(folder).glob('*.pkl')):
            with path.open('rb') as handle:  # local file written by cmd_build or history_replay
                tasks.append(pickle.load(handle))
    overlap = sorted({t['id'] for t in tasks} & banned)
    if overlap:
        raise SystemExit(f'Training tasks overlap evaluation manifests: {overlap[:8]}')
    model = edit_model.fit(tasks, meta={'excluded_manifests': list(args.exclude), 'train_dirs': [str(d) for d in dirs],
                                        'supervision': sorted({t.get('supervision', 'benchmark_patch') for t in tasks})})
    edit_model.save(model, args.out)
    print(json.dumps({k: model.meta[k] for k in ('tasks', 'rows', 'positives')}))


def main(argv=None):
    parser = argparse.ArgumentParser(prog='pathpack-e-train')
    sub = parser.add_subparsers(dest='cmd', required=True)
    build = sub.add_parser('build')
    build.add_argument('--issues', required=True)
    build.add_argument('--out', default='output/pathpack_e/train')
    build.add_argument('--snapshots', default='data/swebench/snapshots')
    build.add_argument('--config', default='configs/pathpack_r.yaml')
    build.add_argument('--v4-config', default='configs/pathpack_v4.json')
    build.add_argument('--cache', default='output/pathpack_r_cache')
    build.add_argument('--history', default='output/pathpack_e/history')
    build.add_argument('--limit', type=int, default=0)
    warm = sub.add_parser('warm')
    warm.add_argument('--issues', required=True)
    warm.add_argument('--snapshots', default='data/swebench/snapshots')
    warm.add_argument('--config', default='configs/pathpack_r.yaml')
    warm.add_argument('--v4-config', default='configs/pathpack_v4.json')
    warm.add_argument('--cache', default='output/pathpack_r_cache')
    warm.add_argument('--history', default='output/pathpack_e/history')
    fit = sub.add_parser('fit')
    fit.add_argument('--train-dir', nargs='+', default=['output/pathpack_e/train'])
    fit.add_argument('--out', default='output/pathpack_e/model.pkl')
    fit.add_argument('--exclude', nargs='+', required=True, help='JSONL manifests that must not overlap training')
    args = parser.parse_args(argv)
    {'build': cmd_build, 'warm': cmd_warm, 'fit': cmd_fit}[args.cmd](args)


if __name__ == '__main__':
    main()

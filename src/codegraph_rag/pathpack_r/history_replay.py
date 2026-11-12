"""History replay: train the edit-location model from a repository's own past commits.

No benchmark label is used. For each repository we take the snapshot of the Verified task with the
earliest ``base_commit`` and replay the commits that precede it: the commit message plays the issue,
the functions the commit touched are the positives. Two guards keep every benchmark fix out:

  1. only ancestors of that earliest ``base_commit`` are read (``git log`` from the snapshot HEAD);
  2. a commit must be older than the earliest ``created_at`` of any Verified task in the repository.

History features of a replayed commit use only strictly older commits (``History.after``).
The code corpus is the snapshot, not the tree at each commit; functions renamed or removed since
then simply yield no positive. Inference never imports this module.
"""
from __future__ import annotations

import argparse
import json
import pickle
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..adaptive_chunker import build_chunk_index
from ..config import V4Config
from ..parser import scan_repository
from ..retrieval import terms
from .config import PathPackRConfig
from .edit_features import build_stage1
from .history import load_history
from .retrieve import RepoIndex, build_pool
from .semantic import SemanticIndex, load_semantic, select_embed_chunks

CHANNELS = ('lexical', 'symbol', 'semantic', 'structural')
_HUNK = re.compile(r'^@@ [^@]* @@\s*(?:async\s+)?(def|class)\s+([A-Za-z_][A-Za-z_0-9]*)')
_TEST = re.compile(r'(^|/)(tests?|testing|conftest\.py)(/|$)|(^|/)test_[^/]*\.py$|_tests?\.py$')
MAX_FILES = 8
MIN_TERMS = 4


def _git(repo, *args, timeout=3600):
    return subprocess.run(['git', '-C', str(repo), *args], check=True, capture_output=True, text=True,
                          timeout=timeout, errors='replace').stdout


def plan(dataset_rows, snapshots):
    """Per repository: the earliest-base snapshot and the cutoff time. Reads no patch."""
    by_repo = {}
    for row in dataset_rows:
        snap = Path(snapshots) / row['instance_id']
        if not snap.is_dir():
            continue
        commit_time = int(_git(snap, 'show', '-s', '--format=%ct', 'HEAD').strip())
        created = row['created_at']
        if not isinstance(created, datetime):
            created = datetime.fromisoformat(str(created).replace('Z', '+00:00'))
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        item = by_repo.setdefault(row['repo'], {'repo': row['repo'], 'commit_time': None, 'cutoff': None, 'tasks': 0})
        item['tasks'] += 1
        if item['commit_time'] is None or commit_time < item['commit_time']:
            item.update(commit_time=commit_time, instance_id=row['instance_id'], base_commit=row['base_commit'])
        stamp = int(created.timestamp())
        if item['cutoff'] is None or stamp < item['cutoff']:
            item['cutoff'] = stamp
    return sorted(by_repo.values(), key=lambda item: item['repo'])


def parse_patch_log(text):
    """Parse ``git log -p -U0 --format=@@@<TAB>hash<TAB>time<TAB>subject<TAB>body<NUL>``."""
    commits = []
    for block in text.split('\n@@@\t') if text else []:
        block = block[4:] if block.startswith('@@@\t') else block
        if '\x00' not in block:
            continue
        head, diff = block.split('\x00', 1)
        parts = head.split('\t', 3)
        if len(parts) < 3:
            continue
        message = parts[2] + ('\n' + parts[3] if len(parts) > 3 else '')
        touched = {}
        path = None
        for line in diff.splitlines():
            if line.startswith('+++ '):
                path = line[6:] if line.startswith('+++ b/') else None
                if path is not None:
                    touched.setdefault(path, set())
            elif line.startswith('@@ ') and path is not None:
                match = _HUNK.match(line)
                if match:
                    touched[path].add(match.group(2))
        commits.append({'hash': parts[0], 'time': int(parts[1]), 'message': message.strip(),
                        'touched': {p: sorted(n) for p, n in touched.items()}})
    return commits


def cmd_fetch(args):
    from ..experiments import load_scoring_dataset
    rows = load_scoring_dataset()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    attributes = out / 'gitattributes'
    attributes.write_text('*.py diff=python\n')
    items = plan(rows, args.snapshots)
    (out / 'plan.json').write_text(json.dumps(items, indent=1) + '\n')
    for item in items:
        dest = out / f'{item["repo"].replace("/", "__")}.commits.json'
        if dest.is_file():
            continue
        snap = Path(args.snapshots) / item['instance_id']
        started = time.perf_counter()
        commits, failed = [], 0
        for skip in range(0, args.commits, args.batch):
            try:
                text = _git(snap, '-c', f'core.attributesFile={attributes.resolve()}', 'log', '-n', str(args.batch),
                            f'--skip={skip}', '--no-merges', f'--before={item["cutoff"] - 1}', '-p', '-U0',
                            '--format=@@@%x09%H%x09%ct%x09%s%x09%b%x00', '--', '*.py', timeout=600)
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
                failed += 1  # a batch whose blobs cannot be fetched is skipped and counted, not hidden
                print(f'  {item["repo"]} batch skip={skip} failed: {str(getattr(exc, "stderr", "") or exc)[-200:]}', flush=True)
                continue
            batch = parse_patch_log(text)
            if not batch:
                break
            commits.extend(c for c in batch if c['time'] < item['cutoff'])
        item['failed_batches'] = failed
        dest.write_text(json.dumps(commits))
        print(f'{item["repo"]}: {len(commits)} commits before cutoff, {time.perf_counter() - started:.0f}s', flush=True)


def label(regions, touched):
    """1 when the chunk is in a touched production file and its function (or class) name was in a hunk header."""
    labels = np.zeros(len(regions), dtype=np.int8)
    for i, region in enumerate(regions):
        names = touched.get(region.chunk.path)
        if not names:
            continue
        parts = (region.chunk.qualname or '').split('.')
        if parts[-1] in names:
            labels[i] = 1
    return labels


def cmd_build(args):
    config = PathPackRConfig.load(args.config)
    v4 = V4Config.load(args.v4_config) if Path(args.v4_config).is_file() else V4Config()
    src = Path(args.logs)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    items = json.loads((src / 'plan.json').read_text())
    sizes = {}
    for item in items:
        path = src / f'{item["repo"].replace("/", "__")}.commits.json'
        item['commits'] = json.loads(path.read_text()) if path.is_file() else []
        sizes[item['repo']] = len(item['commits'])
    for item in items:
        name = item['repo'].replace('/', '__')
        done = out / f'{name}.done'
        if done.is_file() or not item['commits']:
            continue
        snap = Path(args.snapshots) / item['instance_id']
        started = time.perf_counter()
        graph = scan_repository(snap)
        index = RepoIndex(build_chunk_index(snap, graph, v4), graph, snap)
        if len(index.chunks) > args.max_chunks:
            print(f'{item["repo"]}: skipped, {len(index.chunks)} chunks > --max-chunks', flush=True)
            continue
        full = load_semantic(index.chunks, config, cache_dir=args.cache, cache_key=f'replay_{name}')
        if not full.available:
            raise RuntimeError(f'semantic model unavailable: {full.info.get("reason")}')
        position = {cid: i for i, cid in enumerate(full.chunk_ids)}
        history = load_history(snap, cache_dir=args.history, cache_key=f'replay_{name}')
        kept = 0
        for commit in item['commits']:
            if kept >= args.per_repo:
                break
            touched = {p: n for p, n in commit['touched'].items() if n and not _TEST.search(p)}
            production = [p for p in commit['touched'] if not _TEST.search(p)]
            if not touched or len(production) > MAX_FILES or len(set(terms(commit['message']))) < MIN_TERMS:
                continue
            query = commit['message']
            chosen, _report = select_embed_chunks(index, query, config)
            rows_ = [position[c.id] for c in chosen]
            index.semantic = SemanticIndex('available', [c.id for c in chosen], full.matrix[rows_], full.info)
            regions, _r, _q, _l = build_pool(index, query, config, CHANNELS)
            y = label(regions, touched)
            if y.sum() == 0:
                continue
            older = history.after(commit['hash'])
            X, struct = build_stage1(regions, index, older, query=query)
            task = {'id': f'replay:{name}:{commit["hash"][:12]}', 'repo': item['repo'], 'X': X, 'y': y,
                    'struct': struct, 'supervision': 'history_replay', 'regions': len(regions),
                    'positives': int(y.sum()), 'older_commits': len(older.commits)}
            with (out / f'{name}__{commit["hash"][:12]}.pkl').open('wb') as handle:
                pickle.dump(task, handle)
            kept += 1
        done.write_text(json.dumps({'kept': kept, 'commits': len(item['commits']), 'chunks': len(index.chunks)}))
        print(f'{item["repo"]}: kept {kept} of {len(item["commits"])} commits, chunks={len(index.chunks)}, '
              f'{time.perf_counter() - started:.0f}s', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(prog='pathpack-h-replay')
    sub = parser.add_subparsers(dest='cmd', required=True)
    fetch = sub.add_parser('fetch')
    fetch.add_argument('--snapshots', default='data/swebench/snapshots')
    fetch.add_argument('--out', default='output/pathpack_e/replay_logs')
    fetch.add_argument('--commits', type=int, default=300)
    fetch.add_argument('--batch', type=int, default=20)
    build = sub.add_parser('build')
    build.add_argument('--logs', default='output/pathpack_e/replay_logs')
    build.add_argument('--out', default='output/pathpack_e/train_history')
    build.add_argument('--snapshots', default='data/swebench/snapshots')
    build.add_argument('--config', default='configs/pathpack_r.yaml')
    build.add_argument('--v4-config', default='configs/pathpack_v4.json')
    build.add_argument('--cache', default='output/pathpack_r_cache')
    build.add_argument('--history', default='output/pathpack_e/history')
    build.add_argument('--per-repo', type=int, default=120)
    build.add_argument('--max-chunks', type=int, default=30000)
    args = parser.parse_args(argv)
    {'fetch': cmd_fetch, 'build': cmd_build}[args.cmd](args)


if __name__ == '__main__':
    main()

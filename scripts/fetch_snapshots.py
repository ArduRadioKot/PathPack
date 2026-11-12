"""Checkout SWE-bench snapshots that are not already on disk.

A task is scored only after ``git rev-parse HEAD`` matches ``base_commit`` and the
worktree is clean. This script does not compute retrieval metrics.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from codegraph_rag.benchmark import read_jsonl


def _mirror(root, repo):
    name = repo.replace('/', '__') + '.git'
    path = root / name
    if not path.is_dir():
        raise FileNotFoundError(path)
    return path


def checkout(row, mirrors, snapshots):
    destination = (Path(snapshots) / row['instance_id']).resolve()
    if destination.exists():
        head = subprocess.run(['git', '-C', str(destination), 'rev-parse', 'HEAD'], check=True, capture_output=True, text=True).stdout.strip()
        if head != row['base_commit']:
            raise RuntimeError(f'{destination} HEAD {head} != {row["base_commit"]}')
        return {'instance_id': row['instance_id'], 'status': 'present', 'head': head}
    mirror = _mirror(Path(mirrors).resolve(), row['repo'])
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            ['git', '-C', str(mirror), 'worktree', 'add', '--detach', str(destination), row['base_commit']],
            check=True, capture_output=True, text=True,
        )
    except subprocess.CalledProcessError as exc:
        return {
            'instance_id': row['instance_id'], 'status': 'not_checked_out',
            'reason': (exc.stderr or exc.stdout or 'worktree add failed')[-400:],
        }
    head = subprocess.run(['git', '-C', str(destination), 'rev-parse', 'HEAD'], check=True, capture_output=True, text=True).stdout.strip()
    if head != row['base_commit']:
        raise RuntimeError(f'{destination} HEAD {head} != {row["base_commit"]}')
    dirty = subprocess.run(['git', '-C', str(destination), 'status', '--porcelain', '--untracked-files=no'], check=True, capture_output=True, text=True).stdout
    if dirty.strip():
        raise RuntimeError(f'{destination} has tracked changes')
    return {'instance_id': row['instance_id'], 'status': 'checked_out', 'head': head}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--issues', required=True)
    parser.add_argument('--mirrors', default='data/swebench/git_mirrors')
    parser.add_argument('--snapshots', default='data/swebench/snapshots')
    parser.add_argument('--report', required=True)
    args = parser.parse_args()
    rows = read_jsonl(args.issues)
    report = [checkout(row, args.mirrors, args.snapshots) for row in rows]
    Path(args.report).write_text(json.dumps(report, indent=2) + '\n')
    failed = [row for row in report if row['status'] == 'not_checked_out']
    print(json.dumps({'tasks': len(report), 'failed': len(failed)}))
    if failed:
        raise SystemExit(2)


if __name__ == '__main__':
    main()

"""Freeze a larger instance-disjoint split of SWE-bench Verified. No metrics are computed.

Pools:
  diagnostic  the 20 dev tasks, already used many times
  used_test   the other 60 earlier tasks (seen by earlier versions, not independent)
  train_pool  fresh tasks for fitting a ranker or edit-location model
  validation  fresh tasks for choosing the architecture
  final_holdout fresh tasks, looked at once
The frozen 10 + 10 ids from output/pathpack_r_splits/manifest.json stay inside validation and holdout.
Split is stratified by repository. Seed is fixed. Gold patches are read only to write the separate gold file.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='output/pathpack_r_splits_v2')
    ap.add_argument('--seed', type=int, default=46)
    ap.add_argument('--validation', type=int, default=100)
    ap.add_argument('--holdout', type=int, default=150)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f'{out} exists. Frozen splits are not rewritten.')
    out.mkdir(parents=True)

    ds = load_dataset('SWE-bench/SWE-bench_Verified', split='test')
    rows = {r['instance_id']: r for r in ds}
    known = {json.loads(l)['instance_id'] for l in Path('output/benchmark/issues.jsonl').read_text().splitlines() if l.strip()}
    dev = {json.loads(l)['instance_id'] for l in Path('output/benchmark/issues.jsonl').read_text().splitlines()
           if l.strip() and json.loads(l).get('split') == 'dev'}
    old = json.loads(Path('output/pathpack_r_splits/manifest.json').read_text())
    pinned_val, pinned_hold = list(old['validation']), list(old['final_holdout'])
    assert not (set(pinned_val) | set(pinned_hold)) & known

    fresh = sorted(i for i in rows if i not in known)
    rng = random.Random(args.seed)
    by_repo = defaultdict(list)
    for i in fresh:
        if i not in pinned_val and i not in pinned_hold:
            by_repo[rows[i]['repo']].append(i)
    for v in by_repo.values():
        rng.shuffle(v)

    need_val = args.validation - len(pinned_val)
    need_hold = args.holdout - len(pinned_hold)
    pool_n = sum(len(v) for v in by_repo.values())
    validation, holdout, train = list(pinned_val), list(pinned_hold), []
    # proportional quota per repo, remainder to train_pool
    for repo in sorted(by_repo):
        ids = by_repo[repo]
        qv = round(need_val * len(ids) / pool_n)
        qh = round(need_hold * len(ids) / pool_n)
        validation += ids[:qv]
        holdout += ids[qv:qv + qh]
        train += ids[qv + qh:]
    assert not (set(validation) & set(holdout) | set(validation) & set(train) | set(holdout) & set(train))

    def issue(i):
        r = rows[i]
        return {'instance_id': i, 'repo': r['repo'], 'base_commit': r['base_commit'],
                'problem_statement': r['problem_statement']}

    groups = {'validation': validation, 'final_holdout': holdout, 'train_pool': train}
    for name, ids in groups.items():
        (out / f'{name}.jsonl').write_text(''.join(json.dumps(issue(i)) + '\n' for i in sorted(ids)))
    manifest = {
        'seed': args.seed,
        'written_before_any_measurement': True,
        'dataset': 'SWE-bench/SWE-bench_Verified',
        'dataset_size': len(rows),
        'diagnostic_ids': sorted(dev),
        'used_test_ids': sorted(known - dev),
        'pinned_from': 'output/pathpack_r_splits/manifest.json',
        'counts': {k: len(v) for k, v in groups.items()},
        'per_repo': {k: {r: sum(rows[i]['repo'] == r for i in v) for r in sorted({rows[i]['repo'] for i in v})}
                     for k, v in groups.items()},
        **{k: sorted(v) for k, v in groups.items()},
        'repository_disjoint': False,
        'note': 'Instance-disjoint, stratified by repository. Not repository-disjoint. Bootstrap should cluster by repo.',
    }
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=1) + '\n')
    print(json.dumps(manifest['counts']), json.dumps(manifest['per_repo']['final_holdout']))


if __name__ == '__main__':
    main()

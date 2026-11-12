"""Merge PathPack-R retrieval runs over disjoint task sets (or disjoint method sets) into one run. No scoring."""
import json, shutil, sys
from pathlib import Path

out = Path(sys.argv[1]); parts = [Path(p) for p in sys.argv[2:]]
if (out / 'run.json').exists():
    raise SystemExit(f'{out} exists')
(out / 'diagnostics').mkdir(parents=True)
meta = None; issues = {}; preds = {}; methods = []; codes = []
for part in parts:
    m = json.loads((part / 'run.json').read_text())
    codes.append(m['code_sha256'])
    meta = meta or m
    for method in m['methods']:
        if method not in methods:
            methods.append(method)
    done = set()
    for f in sorted(part.glob('*_predictions.jsonl')):
        for line in f.read_text().splitlines():
            row = json.loads(line)
            key = (row['instance_id'], row['method'], row['budget_est'])
            if key in preds:
                raise SystemExit(f'duplicate prediction {key}')
            preds[key] = row; done.add(row['instance_id'])
    for issue in m['issues']:
        if issue['instance_id'] in done:
            issues[issue['instance_id']] = issue
    for f in (part / 'diagnostics').glob('*.json.gz'):
        shutil.copy2(f, out / 'diagnostics' / f.name)
meta = {**meta, 'issues': [issues[k] for k in sorted(issues)], 'methods': methods, 'status': 'retrieval_complete',
        'prediction_count': len(preds), 'merged_from': [str(p) for p in parts], 'part_code_sha256': codes}
rows = [preds[k] for k in sorted(preds, key=lambda k: (k[0], k[1], k[2]))]
(out / 'predictions.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
(out / 'run.json').write_text(json.dumps(meta, indent=1))
print(len(issues), 'tasks', len(rows), 'predictions', methods)

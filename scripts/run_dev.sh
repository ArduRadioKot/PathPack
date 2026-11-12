#!/usr/bin/env bash
set -euo pipefail
# Run from project root, after preparing output/benchmark/issues.jsonl and data/swebench/snapshots/.
ISSUES="${ISSUES:-output/benchmark/issues.jsonl}"
SNAPSHOTS="${SNAPSHOTS:-data/swebench/snapshots}"
GOLD="${GOLD:-output/benchmark/gold_labels.jsonl}"
mkdir -p output/v03_dev reports
for method in bm25 graph-expand pathpack pathpack-v3; do
  for budget in 1024 2048 4096; do
    echo "dev method=$method budget=$budget"
    codegraph retrieve-batch --issues "$ISSUES" --snapshots "$SNAPSHOTS" \
      --split dev --method "$method" --budget "$budget" \
      --out "output/v03_dev/${method}_${budget}.jsonl"
  done
done
codegraph audit --predictions output/v03_dev/*.jsonl --gold "$GOLD" \
  --out reports/audit_dev.json --bootstrap 2000
printf 'Audit: reports/audit_dev.json\n'

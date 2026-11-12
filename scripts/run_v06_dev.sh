#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
.venv/bin/python -m codegraph_rag benchmark \
  --issues output/benchmark/issues.jsonl \
  --snapshots data/swebench/snapshots \
  --out output/v06_dev \
  --split dev \
  --method bm25 graph-expand pathpack-v4 pathpack-v51 pathpack-v6 \
  --budgets 1024 2048 4096 \
  --config configs/pathpack_v4.json \
  --v51-config configs/pathpack_v051.yaml \
  --v6-config configs/pathpack_v06.yaml \
  --v6-ablations
.venv/bin/python -m codegraph_rag score-run --run output/v06_dev
.venv/bin/python -m codegraph_rag audit --run output/v06_dev

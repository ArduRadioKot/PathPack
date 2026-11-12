#!/usr/bin/env bash
# One v0.5.1 dev cycle. Does not overwrite output/v05_dev or output/v04_dev.
set -euo pipefail
cd "$(dirname "$0")/.."
RUN_OUT="${RUN_OUT:-output/v051_dev}"
TASK_PYTHON="${CODEGRAPH_PYTHON:-.venv/bin/python}"
"$TASK_PYTHON" -m codegraph_rag benchmark --split dev \
  --method pathpack-v4 pathpack-v51 --v51-ablations \
  --issues output/benchmark/issues.jsonl --snapshots data/swebench/snapshots \
  --config configs/pathpack_v4.json --v51-config configs/pathpack_v051.yaml \
  --out "$RUN_OUT"
"$TASK_PYTHON" -m codegraph_rag score-run --run "$RUN_OUT"
"$TASK_PYTHON" -m codegraph_rag audit --run "$RUN_OUT"

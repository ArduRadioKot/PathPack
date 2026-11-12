#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
RUN_OUT="${RUN_OUT:-output/v04_dev}"
TASK_PYTHON="${CODEGRAPH_PYTHON:-.venv/bin/python}"
"$TASK_PYTHON" -m codegraph_rag benchmark --split dev --include-ablations \
  --issues output/benchmark/issues.jsonl --snapshots data/swebench/snapshots \
  --config configs/pathpack_v4.json --out "$RUN_OUT"
"$TASK_PYTHON" -m codegraph_rag score-run --run "$RUN_OUT"
"$TASK_PYTHON" -m codegraph_rag audit --run "$RUN_OUT"
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/codegraph-mpl}" "$TASK_PYTHON" -m codegraph_rag visualize-run --run "$RUN_OUT"

#!/usr/bin/env bash
# Dev measurement for PathPack v0.5. Does not read gold until score-run.
set -euo pipefail
cd "$(dirname "$0")/.."
RUN_OUT="${RUN_OUT:-output/v05_dev}"
TASK_PYTHON="${CODEGRAPH_PYTHON:-.venv/bin/python}"
"$TASK_PYTHON" -m codegraph_rag benchmark --split dev \
  --method bm25 graph-expand pathpack-v4 pathpack-v5 \
  --v5-ablations \
  --issues output/benchmark/issues.jsonl --snapshots data/swebench/snapshots \
  --config configs/pathpack_v4.json --v5-config configs/pathpack_v05.yaml \
  --out "$RUN_OUT"
"$TASK_PYTHON" -m codegraph_rag score-run --run "$RUN_OUT"
"$TASK_PYTHON" -m codegraph_rag audit --run "$RUN_OUT"
"$TASK_PYTHON" -m codegraph_rag audit-localization --run output/v04_dev --out output/v05_audit --method pathpack-v4 --budget 2048
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/codegraph-mpl}" "$TASK_PYTHON" -m codegraph_rag visualize-run --run "$RUN_OUT"

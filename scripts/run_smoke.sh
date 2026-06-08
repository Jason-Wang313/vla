#!/usr/bin/env bash
set -euo pipefail

if command -v python.exe >/dev/null 2>&1; then
  PY=(python.exe)
elif command -v python >/dev/null 2>&1; then
  PY=(python)
elif command -v py >/dev/null 2>&1; then
  PY=(py -3)
elif command -v python3 >/dev/null 2>&1; then
  PY=(python3)
else
  echo "No Python interpreter found on PATH" >&2
  exit 127
fi

ROOT_DIR="$("${PY[@]}" - <<'PY'
from pathlib import Path
print(Path(".").resolve())
PY
)"
ROOT_DIR="${ROOT_DIR%$'\r'}"
PATHSEP="$("${PY[@]}" - <<'PY'
import os
print(os.pathsep)
PY
)"
PATHSEP="${PATHSEP%$'\r'}"
export PYTHONPATH="${PYTHONPATH:-}${PATHSEP}$ROOT_DIR${PATHSEP}$ROOT_DIR/src"
export VLA_BON_RESULTS_DIR="${VLA_BON_SMOKE_RESULTS_DIR:-results/smoke}"

"${PY[@]}" -m pytest -q
"${PY[@]}" scripts/run_with_src.py experiments/run_experiments.py --mode smoke --results-dir "$VLA_BON_RESULTS_DIR" --seeds 1 2 --states 5 --candidates 64 --mc-trials 60 --epochs 1
"${PY[@]}" scripts/run_with_src.py scripts/claim_audit.py --results-dir "$VLA_BON_RESULTS_DIR" --fail-on-error

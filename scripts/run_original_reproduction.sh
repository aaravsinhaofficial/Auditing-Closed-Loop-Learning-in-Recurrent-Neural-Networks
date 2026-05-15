#!/usr/bin/env bash
set -euo pipefail
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PWD/.cache/matplotlib}"
mkdir -p "$MPLCONFIGDIR"
python -m closed_loop_repro.analysis.original_artifact_audit --root external/original_artifact --out reproduction_log.md

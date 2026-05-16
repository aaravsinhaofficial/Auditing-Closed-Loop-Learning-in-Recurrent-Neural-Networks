#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PWD/.cache/matplotlib}"
mkdir -p "$MPLCONFIGDIR" logs

python -m closed_loop_repro.training.run_pair --config configs/original/double_integrator_paper_sanity.yaml
python -m closed_loop_repro.analysis.paper_signature_check \
  --result-dir results/raw/original_double_integrator_paper_sanity/seed_0000 \
  --out results/processed/paper_signature_sanity/paper_main_text

python -m closed_loop_repro.training.run_pair --config configs/original/double_integrator_artifact_sanity.yaml
python -m closed_loop_repro.analysis.paper_signature_check \
  --result-dir results/raw/original_double_integrator_artifact_sanity/seed_0000 \
  --out results/processed/paper_signature_sanity/public_nonlinear_notebook

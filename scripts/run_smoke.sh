#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PWD/.cache/matplotlib}"
mkdir -p "$MPLCONFIGDIR"
python -m closed_loop_repro.training.run_pair --config configs/smoke/double_integrator.yaml
python -m closed_loop_repro.sweeps.seed_sweep --config configs/smoke/seed_sweep.yaml
python -m closed_loop_repro.sweeps.robustness_sweep --config configs/smoke/robustness.yaml
python -m closed_loop_repro.sweeps.generalization_sweep --config configs/smoke/generalization.yaml
python -m closed_loop_repro.analysis.make_claim_tables --results results/raw --out results/processed
python -m closed_loop_repro.plotting.make_all_figures --config configs/figures/tmlr.yaml

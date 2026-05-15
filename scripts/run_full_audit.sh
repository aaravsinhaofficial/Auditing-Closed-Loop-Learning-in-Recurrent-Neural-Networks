#!/usr/bin/env bash
set -euo pipefail
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PWD/.cache/matplotlib}"
mkdir -p "$MPLCONFIGDIR"
python -m closed_loop_repro.sweeps.seed_sweep --config configs/original/double_integrator_full.yaml
python -m closed_loop_repro.sweeps.robustness_sweep --config configs/robustness/maximal.yaml
python -m closed_loop_repro.sweeps.generalization_sweep --config configs/generalization/maximal.yaml
python -m closed_loop_repro.analysis.make_claim_tables --results results/raw --out results/processed
python -m closed_loop_repro.plotting.make_all_figures --config configs/figures/tmlr.yaml

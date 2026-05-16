#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PWD/.cache/matplotlib}"
mkdir -p "$MPLCONFIGDIR"

log_step() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')" "$*"
}

log_step "targeted A1 tradeoff sweep start"
python -m closed_loop_repro.sweeps.tradeoff_sweep --config configs/tradeoff/control_penalty_horizon.yaml

log_step "targeted A2 hard path-integration sweep start"
python -m closed_loop_repro.sweeps.generalization_sweep --config configs/generalization/path_integration_hard.yaml

log_step "C2 changepoint reanalysis start"
python -m closed_loop_repro.analysis.stage_changepoints --config configs/stage/changepoint_original.yaml

log_step "timeseries metric recomputation start"
python -m closed_loop_repro.analysis.recompute_timeseries_metrics --results results/raw --out results/processed

log_step "claim tables and figures refresh start"
python -m closed_loop_repro.analysis.make_claim_tables --results results/raw --out results/processed
python -m closed_loop_repro.analysis.supplemental_audit_tables --results results/raw --processed results/processed
python -m closed_loop_repro.plotting.make_all_figures --config configs/figures/tmlr.yaml

log_step "targeted C2/A1/A2 run done"

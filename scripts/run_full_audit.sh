#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PWD/.cache/matplotlib}"
mkdir -p "$MPLCONFIGDIR"

log_step() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')" "$*"
}

log_step "full audit start"
log_step "running original seed sweep"
python -m closed_loop_repro.sweeps.seed_sweep --config configs/original/double_integrator_full.yaml
log_step "running robustness sweep"
python -m closed_loop_repro.sweeps.robustness_sweep --config configs/robustness/maximal.yaml
log_step "running generalization sweep"
python -m closed_loop_repro.sweeps.generalization_sweep --config configs/generalization/maximal.yaml
log_step "running stage changepoint analysis"
python -m closed_loop_repro.analysis.stage_changepoints --config configs/stage/changepoint_original.yaml
log_step "recomputing timeseries metrics"
python -m closed_loop_repro.analysis.recompute_timeseries_metrics --results results/raw --out results/processed
log_step "building claim tables"
python -m closed_loop_repro.analysis.make_claim_tables --results results/raw --out results/processed
log_step "building supplemental audit tables"
python -m closed_loop_repro.analysis.supplemental_audit_tables --results results/raw --processed results/processed
log_step "building figures"
python -m closed_loop_repro.plotting.make_all_figures --config configs/figures/tmlr.yaml
log_step "full audit done"

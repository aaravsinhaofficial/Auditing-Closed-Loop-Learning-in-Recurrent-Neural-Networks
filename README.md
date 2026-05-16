# Auditing Closed-Loop Learning in Recurrent Neural Networks

This repository is a config-driven reproduction and robustness audit for
*Learning Dynamics of RNNs in Closed-Loop Environments* by Yoav Ger and Omri
Barak. The original public artifact is preserved under
`external/original_artifact/`; the reproducibility interface lives in the
`closed_loop_repro` Python package.

## Quick Start

```bash
uv venv --python 3.12
uv pip install -e ".[dev]"
export MPLCONFIGDIR="$PWD/.cache/matplotlib"
pytest
bash scripts/run_smoke.sh
```

The smoke run writes small raw outputs, claim tables, and figures using the
same schema as the full audit.

Configs default to `device: cuda`. On machines without CUDA, the runner falls
back to CPU and records both `requested_device` and `resolved_device` in each
`metrics.json`. To force CPU, set `device: cpu` in the relevant config.

## Full Audit

```bash
bash scripts/run_full_audit.sh
```

Equivalent individual commands:

```bash
python -m closed_loop_repro.sweeps.seed_sweep --config configs/original/double_integrator_full.yaml
python -m closed_loop_repro.sweeps.robustness_sweep --config configs/robustness/maximal.yaml
python -m closed_loop_repro.sweeps.generalization_sweep --config configs/generalization/maximal.yaml
python -m closed_loop_repro.analysis.recompute_timeseries_metrics --results results/raw --out results/processed
python -m closed_loop_repro.analysis.make_claim_tables --results results/raw --out results/processed
python -m closed_loop_repro.analysis.supplemental_audit_tables --results results/raw --processed results/processed
python -m closed_loop_repro.plotting.make_all_figures --config configs/figures/tmlr.yaml
```

For long AWS runs, use `tee` to capture the live console stream:

```bash
mkdir -p logs
bash scripts/run_full_audit.sh 2>&1 | tee logs/full_audit.log
```

The full-audit script uses unbuffered Python output. Sweeps log each run start
and finish, and each paired experiment emits closed-loop/open-loop training
heartbeats every 30 seconds by default. To change the heartbeat cadence, add
`progress_interval_seconds: 60` to a config or under its `training:` block.

## Targeted C2/A1/A2 Follow-Up Runs

The full audit already reproduces the main C1/C3 result. The targeted script
adds experiments for the more delicate stage claim and the two audit questions:

```bash
bash scripts/run_targeted_c2_a1_a2.sh 2>&1 | tee logs/targeted_c2_a1_a2.log
```

Equivalent individual commands:

```bash
# A1: short-vs-long horizon tradeoff over control penalties.
python -m closed_loop_repro.sweeps.tradeoff_sweep \
  --config configs/tradeoff/control_penalty_horizon.yaml

# A2: harder path-integration generalization with partial observation and moving targets.
python -m closed_loop_repro.sweeps.generalization_sweep \
  --config configs/generalization/path_integration_hard.yaml

# C2: segmented-regression changepoint analysis of stage boundaries.
python -m closed_loop_repro.analysis.stage_changepoints \
  --config configs/stage/changepoint_original.yaml

# Refresh timeseries-derived metrics after per-seed outputs are available.
python -m closed_loop_repro.analysis.recompute_timeseries_metrics \
  --results results/raw \
  --out results/processed

# Build run accounting, original-vs-reproduction, A1 component, and C2 sensitivity tables.
python -m closed_loop_repro.analysis.supplemental_audit_tables \
  --results results/raw \
  --processed results/processed
```

The A1 sweep writes `results/raw/tradeoff_control_penalty_horizon/tradeoff_summary.csv`.
The C2 analyzer writes `results/processed/stage_changepoint_summary.csv` and
`results/processed/stage_changepoint_details.csv`. The harder A2 run writes
new `generalization_ring_partial_obs_*` per-seed outputs and updates the
generalization summaries when metrics are recomputed.

## Original Artifact Audit

```bash
bash scripts/run_original_reproduction.sh
```

This creates `reproduction_log.md` with a static inventory, dependency notes,
pickle-load checks, and notebook status. Exact notebook execution can be run
manually with `jupyter nbconvert --execute` from `external/original_artifact/code`.

## Claims and Audit Questions

The package separates original reproduction claims from audit questions:

- C1: closed-loop/open-loop divergence.
- C2: closed-loop stage structure.
- C3: coupled-system stability transition.
- A1: protocol robustness and short-vs-long horizon tradeoff.
- A2: broader generalization across architectures and tasks.

Raw per-run outputs are written under `results/raw/`, processed claim tables
under `results/processed/`, and generated figures under `results/figures/`.
Large experiment outputs are ignored by Git by default.

## Anonymous Supplement Packaging

For double-blind review, build the supplementary archive from tracked files only:

```bash
bash scripts/build_anonymous_supplement.sh
```

This creates `closed_loop_rnn_audit_anonymous_supplement.tar.gz` without
`.git/`, Git remotes, commit history, local AWS logs, virtual environments, or
machine-specific cache paths. Public repository and Software Heritage archive
links are intentionally omitted during anonymous review and should be added only
after de-anonymization or acceptance.

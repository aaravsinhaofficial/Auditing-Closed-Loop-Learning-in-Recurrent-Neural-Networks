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
python -m closed_loop_repro.analysis.make_claim_tables --results results/raw --out results/processed
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

## Original Artifact Audit

```bash
bash scripts/run_original_reproduction.sh
```

This creates `reproduction_log.md` with a static inventory, dependency notes,
pickle-load checks, and notebook status. Exact notebook execution can be run
manually with `jupyter nbconvert --execute` from `external/original_artifact/code`.

## Claims

The package reports claim-level outcomes for:

- C1: closed-loop/open-loop divergence.
- C2: closed-loop stage structure.
- C3: coupled-system stability transition.
- C4: short-term vs long-term tradeoff.
- C5: generalization across architectures and tasks.

Raw per-run outputs are written under `results/raw/`, processed claim tables
under `results/processed/`, and generated figures under `results/figures/`.
Large experiment outputs are ignored by Git by default.

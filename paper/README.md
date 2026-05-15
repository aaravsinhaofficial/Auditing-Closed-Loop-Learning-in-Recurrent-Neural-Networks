# Paper Artifacts

This directory is reserved for TMLR-facing generated material: claim matrices,
figure captions, and paper tables. Run:

```bash
python -m closed_loop_repro.analysis.make_claim_tables --results results/raw --out results/processed
python -m closed_loop_repro.plotting.make_all_figures --config configs/figures/tmlr.yaml
```

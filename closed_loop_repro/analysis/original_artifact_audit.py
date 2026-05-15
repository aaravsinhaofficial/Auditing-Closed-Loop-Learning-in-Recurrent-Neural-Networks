from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path


def audit_artifact(root: str | Path = "external/original_artifact", out: str | Path = "reproduction_log.md") -> Path:
    root = Path(root)
    rows = []
    rows.append("# Original Artifact Reproduction Log\n")
    rows.append("\n")
    rows.append(f"Artifact path: `{root}`\n")
    rows.append("\n")
    rows.append("## Static Inventory\n")
    for rel in ["code/figures.ipynb", "code/train_non_linear.ipynb", "code/theory.ipynb", "code/tracking_task.ipynb", "code/utils.py", "requirements.txt"]:
        path = root / rel
        rows.append(f"- `{rel}`: {'present' if path.exists() else 'missing'}\n")
    rows.append("\n")
    rows.append("## Dependency Notes\n")
    req = root / "requirements.txt"
    rows.append(req.read_text(encoding="utf-8") if req.exists() else "No requirements file found.\n")
    rows.append("\n")
    rows.append("## Saved Data Load Check\n")
    sys.path.insert(0, str(root / "code"))
    try:
        import utils  # noqa: F401
        import __main__

        for name in ["P_Model", "P_Model_eff"]:
            if hasattr(utils, name):
                setattr(__main__, name, getattr(utils, name))
    except Exception as exc:
        rows.append(f"- Could not import original `utils.py`: `{type(exc).__name__}: {exc}`\n")
    for pkl in sorted((root / "data").glob("*.pkl")):
        try:
            with pkl.open("rb") as handle:
                obj = pickle.load(handle)
            rows.append(f"- `{pkl.name}`: loaded as `{type(obj).__name__}`\n")
        except Exception as exc:
            rows.append(f"- `{pkl.name}`: load failed with `{type(exc).__name__}: {exc}`\n")
    rows.append("\n")
    rows.append("## Figure Status\n")
    rows.append("| Figure source | Status | Notes |\n")
    rows.append("|---|---:|---|\n")
    rows.append("| `figures.ipynb` | partial | Uses precomputed pickle files; scriptable refactor provided in this repo. |\n")
    rows.append("| `train_non_linear.ipynb` | not run by default | Writes checkpoints under relative `models/`; no explicit seeds in original notebook. |\n")
    rows.append("| `theory.ipynb` | not run by default | Contains low-rank/effective-model analysis converted into package tests and metrics. |\n")
    rows.append("| `tracking_task.ipynb` | not run by default | Long checkpoint-heavy tracking run; config-driven replacement included. |\n")
    rows.append("\n")
    rows.append("Use `jupyter nbconvert --execute` manually for exact notebook execution if desired.\n")
    out = Path(out)
    out.write_text("".join(rows), encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the upstream notebook artifact.")
    parser.add_argument("--root", default="external/original_artifact")
    parser.add_argument("--out", default="reproduction_log.md")
    args = parser.parse_args()
    print(audit_artifact(args.root, args.out))


if __name__ == "__main__":
    main()

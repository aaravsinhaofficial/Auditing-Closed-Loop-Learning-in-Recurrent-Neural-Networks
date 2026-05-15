from __future__ import annotations

import argparse

from closed_loop_repro.config import load_config
from closed_loop_repro.plotting.figures import make_figures


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate TMLR audit figures.")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    config = load_config(args.config) if args.config else {}
    paths = make_figures(config.get("results", "results/raw"), config.get("processed", "results/processed"), config.get("out", "results/figures"))
    print("\n".join(str(path) for path in paths))


if __name__ == "__main__":
    main()

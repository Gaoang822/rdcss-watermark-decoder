from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pmfa.experiment import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PMFA watermark experiments.")
    parser.add_argument("--quick", action="store_true", help="Run a reduced smoke experiment.")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "full")
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--reuse-cache", action="store_true", help="Reuse previously generated frozen features.")
    parser.add_argument(
        "--reuse-baseline-models",
        action="store_true",
        help="Reuse existing Single-View, PMFA and Decoder-Tail checkpoints while evaluating new adapters.",
    )
    arguments = parser.parse_args()
    result = run_experiment(
        PROJECT_ROOT,
        arguments.data_root,
        arguments.output_dir,
        seed=arguments.seed,
        quick=arguments.quick,
        reuse_cache=arguments.reuse_cache,
        reuse_baseline_models=arguments.reuse_baseline_models,
    )
    print(result["run_info"])
    print(result["summary"].to_string(index=False))


if __name__ == "__main__":
    main()

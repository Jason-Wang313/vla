"""Write optional real-VLA/SmoLVLA adapter status artifacts."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from vla_best_of_n.optional_vla import attempt_smolvla_cpu_inference_probe, write_optional_vla_status


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=os.environ.get("VLA_BON_RESULTS_DIR", "results"))
    parser.add_argument("--cache-root", default=None)
    parser.add_argument("--attempt-inference", action="store_true")
    parser.add_argument("--task", default="put the red mug in the cabinet")
    args = parser.parse_args()
    data = write_optional_vla_status(Path(args.results_dir), args.cache_root)
    print(data["status"])
    print(data["reason"])
    if args.attempt_inference:
        probe = attempt_smolvla_cpu_inference_probe(Path(args.results_dir), args.cache_root, task=args.task)
        print(probe["status"])
        print(probe["reason"])


if __name__ == "__main__":
    main()

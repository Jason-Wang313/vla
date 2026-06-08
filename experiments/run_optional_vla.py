"""Write optional real-VLA/SmoLVLA adapter status artifacts."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from vla_best_of_n.optional_vla import write_optional_vla_status


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=os.environ.get("VLA_BON_RESULTS_DIR", "results"))
    parser.add_argument("--cache-root", default=None)
    args = parser.parse_args()
    data = write_optional_vla_status(Path(args.results_dir), args.cache_root)
    print(data["status"])
    print(data["reason"])


if __name__ == "__main__":
    main()

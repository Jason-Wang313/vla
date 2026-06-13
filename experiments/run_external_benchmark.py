"""Run guarded external benchmark status or short reset/step probes."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from vla_tailguard_audit.external_benchmark import DEFAULT_EXTERNAL_ROOT, DEFAULT_ROBOCASA_ENV_ID, run_external_benchmark


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=os.environ.get("VLA_TAILGUARD_RESULTS_DIR", "results"))
    parser.add_argument("--external-root", default=str(DEFAULT_EXTERNAL_ROOT))
    parser.add_argument("--env-id", default=DEFAULT_ROBOCASA_ENV_ID)
    parser.add_argument("--split", default="pretrain")
    parser.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument("--status-only", action="store_true")
    parser.add_argument("--attempt-robocasa", action="store_true")
    parser.add_argument("--attempt-libero", action="store_true")
    args = parser.parse_args()

    status_only = bool(args.status_only or not args.attempt_robocasa)
    artifact = run_external_benchmark(
        results_dir=Path(args.results_dir),
        external_root=Path(args.external_root),
        env_id=args.env_id,
        split=args.split,
        seeds=args.seeds,
        max_steps=args.max_steps,
        status_only=status_only,
        attempt_robocasa=args.attempt_robocasa,
        attempt_libero=args.attempt_libero,
        timeout_seconds=args.timeout_seconds,
    )
    print(artifact["status"])
    print(artifact["claim_boundary"])


if __name__ == "__main__":
    main()

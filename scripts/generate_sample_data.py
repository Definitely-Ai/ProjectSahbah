from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pose_jitter_lab.sample_data import write_sample_pose


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate deterministic demo pose data.")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "example_pose.csv")
    parser.add_argument("--frames", type=int, default=180)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    path = write_sample_pose(args.out, frames=args.frames, seed=args.seed)
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

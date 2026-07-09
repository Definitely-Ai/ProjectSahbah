from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


JOINTS = [
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
]


def build_sample_pose(frames: int = 180, seed: int = 7) -> pd.DataFrame:
    """Create deterministic paired MediaPipe/stereo pose data for demos."""
    rng = np.random.default_rng(seed)
    rows = []
    trial = "reach_trial_01"

    base_2d = {
        "left_shoulder": np.array([0.41, 0.32]),
        "right_shoulder": np.array([0.61, 0.321]),
        "left_elbow": np.array([0.35, 0.50]),
        "right_elbow": np.array([0.67, 0.50]),
        "left_wrist": np.array([0.30, 0.68]),
        "right_wrist": np.array([0.72, 0.68]),
        "left_hip": np.array([0.44, 0.67]),
        "right_hip": np.array([0.58, 0.67]),
    }
    base_3d = {
        "left_shoulder": np.array([-180.0, 1440.0, 920.0]),
        "right_shoulder": np.array([180.0, 1438.0, 918.0]),
        "left_elbow": np.array([-280.0, 1280.0, 890.0]),
        "right_elbow": np.array([285.0, 1282.0, 886.0]),
        "left_wrist": np.array([-430.0, 1145.0, 850.0]),
        "right_wrist": np.array([430.0, 1146.0, 846.0]),
        "left_hip": np.array([-130.0, 980.0, 940.0]),
        "right_hip": np.array([130.0, 982.0, 936.0]),
    }

    for frame in range(frames):
        phase = frame / frames
        reach = np.sin(phase * np.pi)
        gait = np.sin(phase * np.pi * 4)
        for joint in JOINTS:
            motion_2d = np.zeros(2)
            motion_3d = np.zeros(3)
            if "wrist" in joint:
                side = -1 if joint.startswith("left") else 1
                motion_2d = np.array([side * 0.035 * reach, -0.045 * reach])
                motion_3d = np.array([side * 78.0 * reach, -132.0 * reach, -22.0 * reach])
            elif "elbow" in joint:
                side = -1 if joint.startswith("left") else 1
                motion_2d = np.array([side * 0.018 * reach, -0.02 * reach])
                motion_3d = np.array([side * 42.0 * reach, -74.0 * reach, -10.0 * reach])
            elif "hip" in joint:
                motion_2d = np.array([0.004 * gait, 0.003 * np.cos(phase * np.pi * 3)])
                motion_3d = np.array([8.0 * gait, 5.0 * np.cos(phase * np.pi * 3), 4.0 * gait])

            # MediaPipe jitter is small but relatively visible in normalized space.
            noise_2d = rng.normal(0.0, 0.0016 if "wrist" in joint else 0.0007, size=2)
            # Stereo jitter is in millimeters and includes depth noise.
            noise_3d = rng.normal(0.0, 4.4 if "wrist" in joint else 1.8, size=3)

            point_2d = base_2d[joint] + motion_2d + noise_2d
            point_3d = base_3d[joint] + motion_3d + noise_3d

            rows.append(
                {
                    "source": "mediapipe_lab",
                    "domain": "normalized_2d",
                    "trial": trial,
                    "frame": frame,
                    "time_s": frame / 30.0,
                    "joint": joint,
                    "x": point_2d[0],
                    "y": point_2d[1],
                    "z": np.nan,
                }
            )
            rows.append(
                {
                    "source": "stereo_rig",
                    "domain": "mm_3d",
                    "trial": trial,
                    "frame": frame,
                    "time_s": frame / 30.0,
                    "joint": joint,
                    "x": point_3d[0],
                    "y": point_3d[1],
                    "z": point_3d[2],
                }
            )

    return pd.DataFrame(rows)


def write_sample_pose(path: str | Path, frames: int = 180, seed: int = 7) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    build_sample_pose(frames=frames, seed=seed).to_csv(out, index=False)
    return out

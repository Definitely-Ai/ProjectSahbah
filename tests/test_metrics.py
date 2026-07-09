from __future__ import annotations

import numpy as np
import pandas as pd

from pose_jitter_lab.metrics import compute_jitter, compute_shoulder_widths, summarize_jitter


def test_normalized_jitter_uses_trial_median_shoulder_width() -> None:
    pose = pd.DataFrame(
        [
            row
            for frame, wrist_x in [(0, 0.0), (1, 0.1), (2, 0.2)]
            for row in [
                base_row(frame, "left_shoulder", 0.0, 0.0),
                base_row(frame, "right_shoulder", 0.5, 0.0),
                base_row(frame, "left_wrist", wrist_x, 1.0),
            ]
        ]
    )

    jitter, shoulders = compute_jitter(pose)
    wrist = jitter[jitter["joint"] == "left_wrist"]

    assert np.allclose(shoulders["shoulder_width"], 0.5)
    assert np.allclose(wrist["raw_jitter"], 0.1)
    assert np.allclose(wrist["normalized_jitter"], 0.2)


def test_pair_mode_averages_adjacent_shoulder_widths() -> None:
    pose = pd.DataFrame(
        [
            base_row(0, "left_shoulder", 0.0, 0.0),
            base_row(0, "right_shoulder", 0.5, 0.0),
            base_row(0, "left_wrist", 0.0, 1.0),
            base_row(1, "left_shoulder", 0.0, 0.0),
            base_row(1, "right_shoulder", 1.0, 0.0),
            base_row(1, "left_wrist", 0.5, 1.0),
        ]
    )

    jitter, _ = compute_jitter(pose, shoulder_mode="pair")
    wrist = jitter[jitter["joint"] == "left_wrist"].iloc[0]

    assert wrist["raw_jitter"] == 0.5
    assert wrist["shoulder_scale"] == 0.75
    assert np.isclose(wrist["normalized_jitter"], 2 / 3)


def test_3d_stereo_jitter_uses_xyz_when_present() -> None:
    pose = pd.DataFrame(
        [
            stereo_row(0, "left_shoulder", 0.0, 0.0, 0.0),
            stereo_row(0, "right_shoulder", 400.0, 0.0, 0.0),
            stereo_row(0, "right_wrist", 0.0, 0.0, 0.0),
            stereo_row(1, "left_shoulder", 0.0, 0.0, 0.0),
            stereo_row(1, "right_shoulder", 400.0, 0.0, 0.0),
            stereo_row(1, "right_wrist", 0.0, 30.0, 40.0),
        ]
    )

    jitter, _ = compute_jitter(pose)
    wrist = jitter[jitter["joint"] == "right_wrist"].iloc[0]

    assert wrist["coordinate_dims"] == "x+y+z"
    assert wrist["raw_jitter"] == 50.0
    assert wrist["normalized_jitter"] == 0.125


def test_summary_contains_variance_and_tail_metrics() -> None:
    pose = pd.DataFrame(
        [
            row
            for frame, wrist_x in [(0, 0.0), (1, 0.1), (2, 0.3)]
            for row in [
                base_row(frame, "left_shoulder", 0.0, 0.0),
                base_row(frame, "right_shoulder", 0.5, 0.0),
                base_row(frame, "left_wrist", wrist_x, 1.0),
            ]
        ]
    )

    jitter, _ = compute_jitter(pose)
    summary = summarize_jitter(jitter)
    wrist = summary[summary["joint"] == "left_wrist"].iloc[0]

    assert wrist["frames"] == 2
    assert np.isclose(wrist["normalized_mean"], 0.3)
    assert "normalized_variance" in summary.columns
    assert "normalized_p95" in summary.columns


def test_missing_shoulders_yields_invalid_scale_instead_of_fake_normalization() -> None:
    pose = pd.DataFrame(
        [
            base_row(0, "left_wrist", 0.0, 1.0),
            base_row(1, "left_wrist", 0.1, 1.0),
        ]
    )

    jitter, shoulders = compute_jitter(pose)

    assert shoulders["shoulder_width"].isna().all()
    assert jitter["scale_valid"].eq(False).all()
    assert jitter["normalized_jitter"].isna().all()


def test_compute_shoulder_widths_accepts_custom_names() -> None:
    pose = pd.DataFrame(
        [
            base_row(0, "lshoulder", 0.0, 0.0),
            base_row(0, "rshoulder", 3.0, 4.0),
        ]
    )

    shoulders = compute_shoulder_widths(pose, left_shoulder="LShoulder", right_shoulder="RShoulder")

    assert shoulders.iloc[0]["shoulder_width"] == 5.0


def test_compute_jitter_reports_velocity_when_timestamps_exist() -> None:
    pose = pd.DataFrame(
        [
            {**base_row(0, "left_shoulder", 0.0, 0.0), "time_s": 0.0},
            {**base_row(0, "right_shoulder", 0.5, 0.0), "time_s": 0.0},
            {**base_row(0, "left_wrist", 0.0, 1.0), "time_s": 0.0},
            {**base_row(3, "left_shoulder", 0.0, 0.0), "time_s": 0.5},
            {**base_row(3, "right_shoulder", 0.5, 0.0), "time_s": 0.5},
            {**base_row(3, "left_wrist", 0.1, 1.0), "time_s": 0.5},
        ]
    )

    jitter, _ = compute_jitter(pose)
    wrist = jitter[jitter["joint"] == "left_wrist"].iloc[0]
    summary = summarize_jitter(jitter)

    assert wrist["frame_gap"] == 3
    assert np.isclose(wrist["delta_time_s"], 0.5)
    assert np.isclose(wrist["raw_velocity"], 0.2)
    assert np.isclose(wrist["normalized_velocity"], 0.4)
    assert "normalized_velocity_mean" in summary.columns


def base_row(frame: int, joint: str, x: float, y: float) -> dict[str, object]:
    return {
        "source": "mediapipe_lab",
        "domain": "normalized_2d",
        "trial": "trial_a",
        "frame": frame,
        "joint": joint,
        "x": x,
        "y": y,
    }


def stereo_row(frame: int, joint: str, x: float, y: float, z: float) -> dict[str, object]:
    return {
        "source": "stereo_rig",
        "domain": "mm_3d",
        "trial": "trial_a",
        "frame": frame,
        "joint": joint,
        "x": x,
        "y": y,
        "z": z,
    }

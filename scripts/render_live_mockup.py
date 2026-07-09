from __future__ import annotations

from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pose_jitter_lab.live import draw_dashboard, draw_pose
from pose_jitter_lab.live_metrics import LiveJitterTracker, LiveLandmark


def main() -> int:
    out = ROOT / "reports" / "live" / "mockup-pose-overlay.png"
    out.parent.mkdir(parents=True, exist_ok=True)

    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame[:] = (22, 27, 34)
    for y in range(frame.shape[0]):
        frame[y, :, 0] = min(42, 18 + y // 30)
        frame[y, :, 1] = min(48, 22 + y // 40)
        frame[y, :, 2] = min(58, 30 + y // 45)
    cv2.rectangle(frame, (0, 565), (1280, 720), (36, 42, 48), -1)
    cv2.line(frame, (0, 565), (1280, 565), (65, 75, 82), 2)

    tracker = LiveJitterTracker(trial="mockup_trial", window=45)
    first = fake_landmarks(offset=0.0)
    second = fake_landmarks(offset=0.035)
    tracker.update(frame=0, time_s=0.0, landmarks=first)
    metrics = tracker.update(frame=1, time_s=1 / 30, landmarks=second)

    draw_pose(cv2, frame, second, labels=True, visibility_min=0.35)
    draw_dashboard(
        cv2,
        frame,
        metrics=metrics,
        fps=29.8,
        recording=True,
        rows_written=66,
        record_path=Path("data") / "mockup_trial_pose.csv",
        pose_found=True,
        help_visible=True,
    )
    cv2.imwrite(str(out), frame)
    print(out)
    return 0


def fake_landmarks(offset: float) -> list[LiveLandmark]:
    coords = {
        "nose": (0.50, 0.18),
        "left_eye_inner": (0.49, 0.16),
        "left_eye": (0.48, 0.16),
        "left_eye_outer": (0.47, 0.16),
        "right_eye_inner": (0.51, 0.16),
        "right_eye": (0.52, 0.16),
        "right_eye_outer": (0.53, 0.16),
        "left_ear": (0.45, 0.18),
        "right_ear": (0.55, 0.18),
        "mouth_left": (0.48, 0.22),
        "mouth_right": (0.52, 0.22),
        "left_shoulder": (0.39, 0.32),
        "right_shoulder": (0.61, 0.32),
        "left_elbow": (0.33 - offset * 0.4, 0.48),
        "right_elbow": (0.68 + offset * 0.2, 0.48),
        "left_wrist": (0.29 - offset, 0.65 - offset),
        "right_wrist": (0.72 + offset * 0.5, 0.65),
        "left_pinky": (0.28 - offset, 0.68 - offset),
        "right_pinky": (0.73 + offset * 0.5, 0.68),
        "left_index": (0.30 - offset, 0.68 - offset),
        "right_index": (0.72 + offset * 0.5, 0.68),
        "left_thumb": (0.31 - offset, 0.66 - offset),
        "right_thumb": (0.71 + offset * 0.5, 0.66),
        "left_hip": (0.43, 0.58),
        "right_hip": (0.57, 0.58),
        "left_knee": (0.42, 0.78),
        "right_knee": (0.58, 0.78),
        "left_ankle": (0.40, 0.95),
        "right_ankle": (0.60, 0.95),
        "left_heel": (0.38, 0.98),
        "right_heel": (0.62, 0.98),
        "left_foot_index": (0.43, 0.99),
        "right_foot_index": (0.57, 0.99),
    }
    landmarks = []
    for joint, (x, y) in coords.items():
        landmarks.append(
            LiveLandmark(
                joint=joint,
                x=x,
                y=y,
                z=0.0,
                visibility=0.98,
                presence=0.98,
                x_pixel=int(x * 1280),
                y_pixel=int(y * 720),
            )
        )
    return landmarks


if __name__ == "__main__":
    raise SystemExit(main())

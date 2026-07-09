from __future__ import annotations

import pandas as pd

from pose_jitter_lab.io import load_pose_csv


def test_load_pose_csv_normalizes_common_export_aliases(tmp_path) -> None:
    path = tmp_path / "pose.csv"
    pd.DataFrame(
        [
            {
                "System": "MediaPipe Lab",
                "Frame_Index": 0,
                "Landmark": "Left Shoulder",
                "X_Norm": 0.1,
                "Y_Norm": 0.2,
            },
            {
                "System": "MediaPipe Lab",
                "Frame_Index": 1,
                "Landmark": "Left Shoulder",
                "X_Norm": 0.2,
                "Y_Norm": 0.3,
            },
        ]
    ).to_csv(path, index=False)

    pose = load_pose_csv(path)

    assert pose["source"].tolist() == ["mediapipe_lab", "mediapipe_lab"]
    assert pose["domain"].tolist() == ["mediapipe_lab", "mediapipe_lab"]
    assert pose["trial"].tolist() == ["trial_1", "trial_1"]
    assert pose["joint"].tolist() == ["left_shoulder", "left_shoulder"]
    assert pose["frame"].dtype == "int64"

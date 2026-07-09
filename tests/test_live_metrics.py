from __future__ import annotations

import numpy as np

from pose_jitter_lab.live_metrics import LiveJitterTracker, LiveLandmark


def test_live_tracker_computes_shoulder_normalized_jitter() -> None:
    tracker = LiveJitterTracker(trial="Live Trial", window=10)

    first = tracker.update(frame=0, time_s=0.0, landmarks=landmarks(wrist_x=0.1))
    second = tracker.update(frame=1, time_s=1 / 30, landmarks=landmarks(wrist_x=0.2))

    wrist = next(joint for joint in second.joints if joint.joint == "left_wrist")

    assert first.scale_valid
    assert np.isclose(second.shoulder_scale, 0.5)
    assert np.isclose(wrist.raw_jitter, 0.1)
    assert np.isclose(wrist.normalized_jitter, 0.2)
    assert second.rows[0]["trial"] == "live_trial"


def test_live_tracker_returns_nan_when_shoulders_missing() -> None:
    tracker = LiveJitterTracker(trial="trial")

    tracker.update(frame=0, time_s=0.0, landmarks=[LiveLandmark("left_wrist", 0.1, 0.2)])
    second = tracker.update(frame=1, time_s=1 / 30, landmarks=[LiveLandmark("left_wrist", 0.2, 0.2)])

    wrist = next(joint for joint in second.joints if joint.joint == "left_wrist")

    assert not second.scale_valid
    assert np.isnan(second.shoulder_scale)
    assert np.isnan(wrist.normalized_jitter)


def test_live_tracker_reset_clears_baseline() -> None:
    tracker = LiveJitterTracker(trial="trial")

    tracker.update(frame=0, time_s=0.0, landmarks=landmarks(wrist_x=0.1))
    tracker.reset()
    after_reset = tracker.update(frame=1, time_s=1 / 30, landmarks=landmarks(wrist_x=0.2))
    wrist = next(joint for joint in after_reset.joints if joint.joint == "left_wrist")

    assert np.isnan(wrist.raw_jitter)
    assert np.isnan(wrist.normalized_jitter)


def test_live_tracker_reports_time_normalized_velocity() -> None:
    tracker = LiveJitterTracker(trial="trial")

    tracker.update(frame=0, time_s=0.0, landmarks=landmarks(wrist_x=0.1))
    second = tracker.update(frame=1, time_s=0.5, landmarks=landmarks(wrist_x=0.2))
    wrist = next(joint for joint in second.joints if joint.joint == "left_wrist")

    assert np.isclose(wrist.delta_time_s, 0.5)
    assert np.isclose(wrist.raw_velocity, 0.2)
    assert np.isclose(wrist.normalized_velocity, 0.4)
    assert np.isclose(second.rows[0]["delta_time_s"], 0.5)


def test_live_tracker_rejects_low_visibility_landmarks() -> None:
    tracker = LiveJitterTracker(trial="trial", visibility_min=0.35)

    tracker.update(frame=0, time_s=0.0, landmarks=landmarks(wrist_x=0.1))
    low_confidence = landmarks(wrist_x=0.2)
    low_confidence[2] = LiveLandmark("left_wrist", 0.2, 0.5, visibility=0.1)
    second = tracker.update(frame=1, time_s=1 / 30, landmarks=low_confidence)
    wrist = next(joint for joint in second.joints if joint.joint == "left_wrist")

    assert not wrist.landmark_valid
    assert wrist.invalid_reason == "low_visibility"
    assert np.isnan(wrist.raw_jitter)
    assert second.invalid_landmarks == 1


def test_live_tracker_does_not_bridge_across_invalid_dropout() -> None:
    tracker = LiveJitterTracker(trial="trial", visibility_min=0.35)

    tracker.update(frame=0, time_s=0.0, landmarks=landmarks(wrist_x=0.1))
    dropout = landmarks(wrist_x=0.2)
    dropout[2] = LiveLandmark("left_wrist", 0.2, 0.5, visibility=0.1)
    tracker.update(frame=1, time_s=1 / 30, landmarks=dropout)
    after_dropout = tracker.update(frame=2, time_s=2 / 30, landmarks=landmarks(wrist_x=0.3))
    wrist = next(joint for joint in after_dropout.joints if joint.joint == "left_wrist")

    assert np.isnan(wrist.raw_jitter)
    assert np.isnan(wrist.normalized_jitter)


def landmarks(wrist_x: float) -> list[LiveLandmark]:
    return [
        LiveLandmark("left_shoulder", 0.0, 0.0, visibility=1.0),
        LiveLandmark("right_shoulder", 0.5, 0.0, visibility=1.0),
        LiveLandmark("left_wrist", wrist_x, 0.5, visibility=1.0),
    ]

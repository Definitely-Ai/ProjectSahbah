from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import sqrt
from statistics import median

import numpy as np

from .io import normalize_label


@dataclass(frozen=True)
class LiveLandmark:
    joint: str
    x: float
    y: float
    z: float | None = None
    visibility: float | None = None
    presence: float | None = None
    x_pixel: int | None = None
    y_pixel: int | None = None

    def coords(self, include_z: bool = False) -> tuple[float, ...]:
        if include_z and self.z is not None and not np.isnan(self.z):
            return (self.x, self.y, float(self.z))
        return (self.x, self.y)


@dataclass(frozen=True)
class LiveJointMetric:
    joint: str
    raw_jitter: float
    normalized_jitter: float
    delta_time_s: float
    raw_velocity: float
    normalized_velocity: float
    rolling_mean: float
    rolling_std: float
    landmark_valid: bool
    invalid_reason: str


@dataclass(frozen=True)
class LiveFrameMetrics:
    frame: int
    shoulder_width: float
    shoulder_scale: float
    scale_valid: bool
    joints: list[LiveJointMetric]
    rows: list[dict[str, object]]
    valid_landmarks: int
    invalid_landmarks: int

    def top_joints(self, limit: int = 5) -> list[LiveJointMetric]:
        return sorted(
            [joint for joint in self.joints if not np.isnan(joint.normalized_jitter)],
            key=lambda joint: joint.rolling_mean,
            reverse=True,
        )[:limit]


class LiveJitterTracker:
    """Rolling shoulder-normalized jitter tracker for real-time pose streams."""

    def __init__(
        self,
        *,
        trial: str,
        source: str = "mediapipe_live",
        domain: str = "normalized_2d",
        left_shoulder: str = "left_shoulder",
        right_shoulder: str = "right_shoulder",
        include_z: bool = False,
        window: int = 45,
        min_shoulder_width: float = 1e-9,
        visibility_min: float = 0.35,
        presence_min: float = 0.35,
    ) -> None:
        self.trial = normalize_label(trial)
        self.source = normalize_label(source)
        self.domain = normalize_label(domain)
        self.left_shoulder = normalize_label(left_shoulder)
        self.right_shoulder = normalize_label(right_shoulder)
        self.include_z = include_z
        self.min_shoulder_width = min_shoulder_width
        self.visibility_min = visibility_min
        self.presence_min = presence_min
        self.shoulder_widths: deque[float] = deque(maxlen=window)
        self.rolling_by_joint: dict[str, deque[float]] = {}
        self.previous: dict[str, LiveLandmark] = {}
        self.previous_time_s: dict[str, float] = {}
        self.window = window
        self.scale_mode = "rolling_median_shoulder_width"

    def reset(self) -> None:
        self.shoulder_widths.clear()
        self.rolling_by_joint.clear()
        self.previous.clear()
        self.previous_time_s.clear()

    def update(
        self,
        *,
        frame: int,
        time_s: float,
        landmarks: list[LiveLandmark],
    ) -> LiveFrameMetrics:
        current = {normalize_label(landmark.joint): landmark for landmark in landmarks}
        valid_current = {
            joint: landmark
            for joint, landmark in current.items()
            if not self._invalid_reason(landmark)
        }
        shoulder_width = self._shoulder_width(valid_current)
        if not np.isnan(shoulder_width) and shoulder_width > self.min_shoulder_width:
            self.shoulder_widths.append(shoulder_width)
        shoulder_scale = float(median(self.shoulder_widths)) if self.shoulder_widths else np.nan
        scale_valid = bool(not np.isnan(shoulder_scale) and shoulder_scale > self.min_shoulder_width)

        joint_metrics: list[LiveJointMetric] = []
        rows: list[dict[str, object]] = []
        for joint, landmark in sorted(current.items()):
            invalid_reason = self._invalid_reason(landmark)
            landmark_valid = not invalid_reason
            previous = self.previous.get(joint) if landmark_valid else None
            raw_jitter = np.nan
            normalized_jitter = np.nan
            delta_time_s = np.nan
            raw_velocity = np.nan
            normalized_velocity = np.nan
            if previous is not None:
                raw_jitter = euclidean(previous.coords(self.include_z), landmark.coords(self.include_z))
                if scale_valid:
                    normalized_jitter = raw_jitter / shoulder_scale
                previous_time_s = self.previous_time_s.get(joint, np.nan)
                if not np.isnan(previous_time_s):
                    delta_time_s = float(time_s - previous_time_s)
                if not np.isnan(delta_time_s) and delta_time_s > 0:
                    raw_velocity = raw_jitter / delta_time_s
                    if scale_valid:
                        normalized_velocity = normalized_jitter / delta_time_s

            rolling = self.rolling_by_joint.setdefault(joint, deque(maxlen=self.window))
            if not np.isnan(normalized_jitter):
                rolling.append(float(normalized_jitter))
            rolling_values = np.array(rolling, dtype=float)
            rolling_mean = float(np.mean(rolling_values)) if rolling_values.size else np.nan
            rolling_std = float(np.std(rolling_values, ddof=1)) if rolling_values.size > 1 else np.nan

            joint_metrics.append(
                LiveJointMetric(
                    joint=joint,
                    raw_jitter=float(raw_jitter),
                    normalized_jitter=float(normalized_jitter),
                    delta_time_s=float(delta_time_s),
                    raw_velocity=float(raw_velocity),
                    normalized_velocity=float(normalized_velocity),
                    rolling_mean=rolling_mean,
                    rolling_std=rolling_std,
                    landmark_valid=landmark_valid,
                    invalid_reason=invalid_reason,
                )
            )
            rows.append(
                {
                    "source": self.source,
                    "domain": self.domain,
                    "trial": self.trial,
                    "frame": frame,
                    "time_s": time_s,
                    "joint": joint,
                    "x": landmark.x,
                    "y": landmark.y,
                    "z": landmark.z,
                    "visibility": landmark.visibility,
                    "presence": landmark.presence,
                    "x_pixel": landmark.x_pixel,
                    "y_pixel": landmark.y_pixel,
                    "raw_jitter": raw_jitter,
                    "normalized_jitter": normalized_jitter,
                    "delta_time_s": delta_time_s,
                    "raw_velocity": raw_velocity,
                    "normalized_velocity": normalized_velocity,
                    "rolling_mean_jitter": rolling_mean,
                    "rolling_std_jitter": rolling_std,
                    "shoulder_width": shoulder_width,
                    "shoulder_scale": shoulder_scale,
                    "scale_valid": scale_valid,
                    "scale_mode": self.scale_mode,
                    "landmark_valid": landmark_valid,
                    "invalid_reason": invalid_reason,
                }
            )

        self.previous = valid_current
        self.previous_time_s = {joint: float(time_s) for joint in valid_current}
        return LiveFrameMetrics(
            frame=frame,
            shoulder_width=shoulder_width,
            shoulder_scale=shoulder_scale,
            scale_valid=scale_valid,
            joints=joint_metrics,
            rows=rows,
            valid_landmarks=len(valid_current),
            invalid_landmarks=len(current) - len(valid_current),
        )

    def _shoulder_width(self, current: dict[str, LiveLandmark]) -> float:
        left = current.get(self.left_shoulder)
        right = current.get(self.right_shoulder)
        if left is None or right is None:
            return np.nan
        return euclidean(left.coords(self.include_z), right.coords(self.include_z))

    def _invalid_reason(self, landmark: LiveLandmark) -> str:
        coords = landmark.coords(self.include_z)
        if any(np.isnan(float(value)) for value in coords):
            return "missing_coordinate"
        if _below_threshold(landmark.visibility, self.visibility_min):
            return "low_visibility"
        if _below_threshold(landmark.presence, self.presence_min):
            return "low_presence"
        return ""


def euclidean(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    return sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(left, right)))


def _below_threshold(value: float | None, threshold: float) -> bool:
    if value is None or np.isnan(value):
        return False
    return float(value) < threshold

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import time
from urllib.request import urlretrieve

import numpy as np

from .live_metrics import LiveJitterTracker, LiveLandmark, LiveFrameMetrics


POSE_MODEL_LITE_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
DEFAULT_MODEL_PATH = Path("models") / "pose_landmarker_lite.task"

LANDMARK_NAMES = [
    "nose",
    "left_eye_inner",
    "left_eye",
    "left_eye_outer",
    "right_eye_inner",
    "right_eye",
    "right_eye_outer",
    "left_ear",
    "right_ear",
    "mouth_left",
    "mouth_right",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_pinky",
    "right_pinky",
    "left_index",
    "right_index",
    "left_thumb",
    "right_thumb",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
]

POSE_CONNECTIONS = [
    (11, 12),
    (11, 13),
    (13, 15),
    (15, 17),
    (15, 19),
    (15, 21),
    (12, 14),
    (14, 16),
    (16, 18),
    (16, 20),
    (16, 22),
    (11, 23),
    (12, 24),
    (23, 24),
    (23, 25),
    (25, 27),
    (27, 29),
    (29, 31),
    (24, 26),
    (26, 28),
    (28, 30),
    (30, 32),
    (0, 2),
    (0, 5),
    (2, 7),
    (5, 8),
]

LABEL_JOINTS = {
    "nose",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
}

BODY_JITTER_JOINTS = {
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
}

CSV_FIELDS = [
    "source",
    "domain",
    "trial",
    "frame",
    "time_s",
    "joint",
    "x",
    "y",
    "z",
    "visibility",
    "presence",
    "x_pixel",
    "y_pixel",
    "raw_jitter",
    "normalized_jitter",
    "delta_time_s",
    "raw_velocity",
    "normalized_velocity",
    "rolling_mean_jitter",
    "rolling_std_jitter",
    "shoulder_width",
    "shoulder_scale",
    "scale_valid",
    "scale_mode",
    "landmark_valid",
    "invalid_reason",
]


@dataclass
class LiveConfig:
    camera: int = 0
    model: Path = DEFAULT_MODEL_PATH
    record: Path | None = None
    auto_record: bool = False
    trial: str | None = None
    width: int = 1280
    height: int = 720
    fps: int = 30
    backend: str = "dshow"
    visibility_min: float = 0.35
    presence_min: float = 0.35
    jitter_window: int = 45
    include_z: bool = False
    max_frames: int | None = None
    headless: bool = False
    mirror_display: bool = False
    save_preview: Path | None = None


class LiveCsvRecorder:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        self.writer.writeheader()
        self.rows_written = 0

    def write_rows(self, rows: list[dict[str, object]]) -> None:
        for row in rows:
            self.writer.writerow(row)
            self.rows_written += 1
        self.handle.flush()

    def close(self) -> None:
        self.handle.close()


def run_camera_list(max_index: int = 8, backend: str = "dshow") -> int:
    cv2 = import_cv2()
    backend_id = cv_backend(cv2, backend)
    print("camera opened read width height fps")
    for index in range(max_index):
        cap = cv2.VideoCapture(index, backend_id)
        opened = cap.isOpened()
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if opened else 0
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if opened else 0
        fps = cap.get(cv2.CAP_PROP_FPS) if opened else 0
        ok = False
        if opened:
            ok, _ = cap.read()
        cap.release()
        print(f"{index:>6} {str(opened):>6} {str(ok):>4} {width:>5} {height:>6} {fps:>5.1f}")
    return 0


def run_live(config: LiveConfig) -> int:
    cv2 = import_cv2()
    mp = import_mediapipe()
    ensure_pose_model(config.model)

    trial = config.trial or f"live_trial_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    record_path = config.record or Path("data") / f"{trial}_pose.csv"
    tracker = LiveJitterTracker(
        trial=trial,
        include_z=config.include_z,
        window=config.jitter_window,
        visibility_min=config.visibility_min,
        presence_min=config.presence_min,
    )
    recorder = LiveCsvRecorder(record_path) if config.auto_record else None
    recording = config.auto_record

    cap = open_capture(cv2, config)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {config.camera}. Run `python pose_jitter.py cameras`.")

    landmarker = create_pose_landmarker(mp, config.model)
    start = time.perf_counter()
    last_timestamp_ms = -1
    last_frame_time = start
    frame_index = 0
    paused = False
    labels = True
    help_visible = False
    last_metrics: LiveFrameMetrics | None = None
    preview_saved = False

    try:
        while True:
            if not paused:
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError("Camera opened but frame capture failed.")
                if config.mirror_display:
                    frame = cv2.flip(frame, 1)

                now = time.perf_counter()
                fps_value = 1.0 / max(now - last_frame_time, 1e-9)
                last_frame_time = now
                timestamp_ms = max(int((now - start) * 1000), last_timestamp_ms + 1)
                last_timestamp_ms = timestamp_ms

                result = detect_pose(mp, landmarker, frame, timestamp_ms)
                landmarks = extract_landmarks(result, frame.shape)
                if landmarks:
                    last_metrics = tracker.update(
                        frame=frame_index,
                        time_s=now - start,
                        landmarks=landmarks,
                    )
                    draw_pose(cv2, frame, landmarks, labels=labels, visibility_min=config.visibility_min)
                    if recording and recorder is not None:
                        recorder.write_rows(last_metrics.rows)
                draw_dashboard(
                    cv2,
                    frame,
                    metrics=last_metrics,
                    fps=fps_value,
                    recording=recording,
                    rows_written=recorder.rows_written if recorder else 0,
                    record_path=record_path,
                    pose_found=bool(landmarks),
                    help_visible=help_visible,
                )

                if config.save_preview and not preview_saved:
                    config.save_preview.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(config.save_preview), frame)
                    preview_saved = True

                frame_index += 1

            if not config.headless:
                cv2.imshow("Pose Jitter Live Lab", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("r"):
                    if recorder is None:
                        recorder = LiveCsvRecorder(record_path)
                    recording = not recording
                elif key == ord("c"):
                    tracker.reset()
                    last_metrics = None
                elif key == ord("l"):
                    labels = not labels
                elif key == ord("h"):
                    help_visible = not help_visible
                elif key == ord("p"):
                    paused = not paused
                elif key == ord("s"):
                    snapshot = Path("reports") / "live" / f"snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                    snapshot.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(snapshot), frame)
                    print(f"Saved snapshot: {snapshot}")

            if config.max_frames is not None and frame_index >= config.max_frames:
                break
            if config.headless and config.max_frames is None:
                break
    finally:
        cap.release()
        landmarker.close()
        if recorder is not None:
            recorder.close()
            print(f"Wrote live pose CSV: {record_path}")
            print(f"Rows written: {recorder.rows_written}")
        if not config.headless:
            cv2.destroyAllWindows()
    return 0


def run_stereo_preview(
    *,
    left_camera: int,
    right_camera: int,
    model: Path = DEFAULT_MODEL_PATH,
    backend: str = "dshow",
    max_frames: int | None = None,
) -> int:
    """Preview two webcams side by side with MediaPipe landmarks.

    This is intentionally a preview, not calibrated millimeter reconstruction.
    For mm output, the cameras need stereo calibration first.
    """
    cv2 = import_cv2()
    mp = import_mediapipe()
    ensure_pose_model(model)
    backend_id = cv_backend(cv2, backend)
    left = cv2.VideoCapture(left_camera, backend_id)
    right = cv2.VideoCapture(right_camera, backend_id)
    if not left.isOpened() or not right.isOpened():
        raise RuntimeError("Could not open both stereo cameras. Run `python pose_jitter.py cameras`.")

    left_landmarker = create_pose_landmarker(mp, model)
    right_landmarker = create_pose_landmarker(mp, model)
    start = time.perf_counter()
    frame_index = 0
    timestamp_ms = 0
    try:
        while True:
            ok_left, left_frame = left.read()
            ok_right, right_frame = right.read()
            if not ok_left or not ok_right:
                raise RuntimeError("Stereo camera frame capture failed.")
            now_ms = max(int((time.perf_counter() - start) * 1000), timestamp_ms + 1)
            timestamp_ms = now_ms
            left_landmarks = extract_landmarks(detect_pose(mp, left_landmarker, left_frame, timestamp_ms), left_frame.shape)
            right_landmarks = extract_landmarks(detect_pose(mp, right_landmarker, right_frame, timestamp_ms), right_frame.shape)
            draw_pose(cv2, left_frame, left_landmarks, labels=False, visibility_min=0.35)
            draw_pose(cv2, right_frame, right_landmarks, labels=False, visibility_min=0.35)
            cv2.putText(left_frame, f"Left camera {left_camera}", (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(right_frame, f"Right camera {right_camera}", (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            view = np.hstack([resize_to_height(cv2, left_frame, 540), resize_to_height(cv2, right_frame, 540)])
            cv2.imshow("Stereo Pose Preview - calibrated mm requires stereo calibration", view)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            frame_index += 1
            if max_frames is not None and frame_index >= max_frames:
                break
    finally:
        left.release()
        right.release()
        left_landmarker.close()
        right_landmarker.close()
        cv2.destroyAllWindows()
    return 0


def import_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for live camera mode. Use `.\\.venv312\\Scripts\\python.exe pose_jitter.py live`.") from exc
    return cv2


def import_mediapipe():
    try:
        import mediapipe as mp
    except ImportError as exc:
        raise RuntimeError("MediaPipe is required for live pose mode. Use `.\\.venv312\\Scripts\\python.exe pose_jitter.py live`.") from exc
    if not hasattr(mp, "tasks"):
        raise RuntimeError("This tool expects the MediaPipe Tasks API. Install a current mediapipe package in Python 3.12.")
    return mp


def ensure_pose_model(path: Path) -> None:
    if path.exists() and path.stat().st_size > 1_000_000:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading MediaPipe pose model: {path}")
    urlretrieve(POSE_MODEL_LITE_URL, path)


def cv_backend(cv2, name: str) -> int:
    name = name.lower()
    if name == "dshow":
        return cv2.CAP_DSHOW
    if name == "msmf":
        return cv2.CAP_MSMF
    if name == "any":
        return cv2.CAP_ANY
    raise ValueError("backend must be one of: dshow, msmf, any")


def open_capture(cv2, config: LiveConfig):
    cap = cv2.VideoCapture(config.camera, cv_backend(cv2, config.backend))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.height)
    cap.set(cv2.CAP_PROP_FPS, config.fps)
    return cap


def create_pose_landmarker(mp, model_path: Path):
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp.tasks.vision.PoseLandmarker.create_from_options(options)


def detect_pose(mp, landmarker, frame, timestamp_ms: int):
    cv2 = import_cv2()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
    return landmarker.detect_for_video(mp_image, timestamp_ms)


def extract_landmarks(result, frame_shape) -> list[LiveLandmark]:
    if not result.pose_landmarks:
        return []
    height, width = frame_shape[:2]
    landmarks = []
    for index, landmark in enumerate(result.pose_landmarks[0]):
        name = LANDMARK_NAMES[index] if index < len(LANDMARK_NAMES) else f"landmark_{index}"
        x_pixel = int(round(landmark.x * width))
        y_pixel = int(round(landmark.y * height))
        landmarks.append(
            LiveLandmark(
                joint=name,
                x=float(landmark.x),
                y=float(landmark.y),
                z=float(landmark.z),
                visibility=float(getattr(landmark, "visibility", np.nan)),
                presence=float(getattr(landmark, "presence", np.nan)),
                x_pixel=x_pixel,
                y_pixel=y_pixel,
            )
        )
    return landmarks


def draw_pose(cv2, frame, landmarks: list[LiveLandmark], *, labels: bool, visibility_min: float) -> None:
    by_index = {index: landmark for index, landmark in enumerate(landmarks)}
    for start, end in POSE_CONNECTIONS:
        a = by_index.get(start)
        b = by_index.get(end)
        if not visible(a, visibility_min) or not visible(b, visibility_min):
            continue
        cv2.line(frame, (a.x_pixel, a.y_pixel), (b.x_pixel, b.y_pixel), (20, 210, 190), 3, cv2.LINE_AA)

    for landmark in landmarks:
        if not visible(landmark, visibility_min):
            continue
        color = (26, 92, 245) if landmark.joint in {"left_wrist", "right_wrist", "left_ankle", "right_ankle"} else (255, 255, 255)
        cv2.circle(frame, (landmark.x_pixel, landmark.y_pixel), 5, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(frame, (landmark.x_pixel, landmark.y_pixel), 4, color, -1, cv2.LINE_AA)
        if labels and landmark.joint in LABEL_JOINTS:
            draw_text(cv2, frame, short_label(landmark.joint), (landmark.x_pixel + 6, landmark.y_pixel - 6), 0.42)


def visible(landmark: LiveLandmark | None, threshold: float) -> bool:
    if landmark is None:
        return False
    return landmark.visibility is None or np.isnan(landmark.visibility) or landmark.visibility >= threshold


def draw_dashboard(
    cv2,
    frame,
    *,
    metrics: LiveFrameMetrics | None,
    fps: float,
    recording: bool,
    rows_written: int,
    record_path: Path,
    pose_found: bool,
    help_visible: bool,
) -> None:
    height, width = frame.shape[:2]
    status = "POSE LOCK" if pose_found else "NO POSE"
    rec = "REC" if recording else "idle"
    lines = [
        f"Pose Jitter Live Lab  |  {status}",
        f"FPS {fps:5.1f}  |  recording {rec}  |  rows {rows_written}",
    ]
    if metrics is not None:
        lines.append(f"shoulder width {metrics.shoulder_width:.5f}  |  scale {metrics.shoulder_scale:.5f}")
        lines.append(f"valid landmarks {metrics.valid_landmarks:02d}  |  rejected {metrics.invalid_landmarks:02d}")
        top = dashboard_top_joints(metrics)
        if top:
            lines.append("top rolling normalized jitter:")
            for joint in top:
                lines.append(f"  {joint.joint:<17} {joint.rolling_mean:8.5f}")
        else:
            lines.append("move for one more frame to compute jitter")
    else:
        lines.append("stand in frame so MediaPipe can lock onto the body")
        lines.append("shoulders must be visible for normalized jitter")

    if help_visible:
        lines.extend(
            [
                "",
                "keys: q quit | r record | c reset | l labels",
                "      s snapshot | p pause | h help",
                f"csv: {record_path}",
            ]
        )
    else:
        lines.append("press h for controls")

    panel_width = min(455, width - 24)
    panel_height = min(height - 24, max(150, 34 + len(lines) * 24))
    overlay = frame.copy()
    cv2.rectangle(overlay, (12, 12), (12 + panel_width, 12 + panel_height), (12, 18, 28), -1)
    cv2.addWeighted(overlay, 0.76, frame, 0.24, 0, frame)
    cv2.rectangle(frame, (12, 12), (12 + panel_width, 12 + panel_height), (20, 210, 190), 2)

    y = 40
    for index, line in enumerate(lines):
        color = (255, 255, 255)
        if index == 0:
            color = (20, 240, 210) if pose_found else (50, 170, 255)
        draw_text(cv2, frame, line, (28, y), 0.56 if index == 0 else 0.48, color=color)
        y += 24


def dashboard_top_joints(metrics: LiveFrameMetrics, limit: int = 4):
    major = [
        joint
        for joint in metrics.joints
        if joint.joint in BODY_JITTER_JOINTS and not np.isnan(joint.normalized_jitter)
    ]
    if not major:
        return metrics.top_joints(limit)
    return sorted(major, key=lambda joint: joint.rolling_mean, reverse=True)[:limit]


def draw_text(cv2, frame, text: str, origin: tuple[int, int], scale: float, color=(255, 255, 255)) -> None:
    x, y = origin
    cv2.putText(frame, text, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def short_label(joint: str) -> str:
    return joint.replace("left_", "L ").replace("right_", "R ").replace("_", " ")


def resize_to_height(cv2, frame, height: int):
    current_height, current_width = frame.shape[:2]
    scale = height / current_height
    return cv2.resize(frame, (int(current_width * scale), height), interpolation=cv2.INTER_AREA)

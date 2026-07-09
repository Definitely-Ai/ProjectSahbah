# Research-Backed Improvement Plan

This document turns the current code audit and external research into a practical thesis software roadmap. The goal is not just a nicer demo. The goal is a defensible measurement tool for comparing MediaPipe normalized pose jitter with calibrated stereo/OpenCV pose jitter.

## Current State

The project already has the right core idea:

```text
normalized_jitter = frame_to_frame_euclidean_distance / same_domain_shoulder_width
```

That avoids inventing a fake conversion from MediaPipe normalized image coordinates to stereo millimeters. The current application also gives the student the critical development loop: live webcam landmarks, shoulder-normalized rolling jitter, CSV recording, offline analysis, and an HTML report.

The biggest remaining risks are measurement validity risks rather than UI risks:

- Frame-to-frame jitter can be distorted by variable FPS unless per-second velocity is reported.
- Low-confidence landmarks can create false jitter spikes if they are not excluded or labeled.
- Raw and filtered data need to be separated so postprocessing does not hide the original measurement.
- Stereo millimeter output is not defensible until the cameras are calibrated and rectified.
- The model file, MediaPipe version, camera settings, and calibration files need provenance in every trial.

## Improvements Implemented From This Research Pass

1. Timestamp-aware jitter velocity.
   - Offline jitter rows now include `frame_gap`, `delta_time_s`, `raw_velocity`, and `normalized_velocity`.
   - Summaries now include velocity columns when timestamps exist.
   - Live CSV rows now include `delta_time_s`, `raw_velocity`, and `normalized_velocity`.

2. Live landmark confidence gating.
   - Live metrics now reject landmarks below `--visibility-min` or `--presence-min`.
   - Invalid landmarks are written with `landmark_valid` and `invalid_reason`.
   - The tracker does not bridge jitter across an invalid dropout frame.

These changes make the data easier to defend when webcams miss frames, run at uneven FPS, or briefly lose a joint.

## Highest-Impact Roadmap

### P0 - Measurement Correctness

1. Add experiment metadata sidecars.
   - Write `trial_metadata.json` next to every live CSV.
   - Include subject/session/trial labels, camera index, requested and actual FPS, resolution, OpenCV backend, model path, model hash, MediaPipe version, OpenCV version, OS, timestamp, and whether mirroring was enabled.
   - Code targets: `src/pose_jitter_lab/live.py`, `src/pose_jitter_lab/cli.py`.

2. Pin and verify the MediaPipe model.
   - Replace the mutable `latest` model URL with a pinned release URL or store a required SHA-256 hash.
   - Record the model hash in the CSV metadata.
   - Code target: `ensure_pose_model()` in `src/pose_jitter_lab/live.py`.

3. Protect trial recordings from accidental overwrite.
   - Add `--overwrite` or auto-suffix duplicate recording paths.
   - Code targets: `LiveCsvRecorder`, CLI args.

4. Add a baseline/no-motion protocol.
   - Add a `baseline` command or `--baseline-seconds`.
   - Record a still subject first to estimate camera/model jitter floor.
   - Report baseline-subtracted and raw metrics separately.

5. Preserve raw-vs-filtered traces.
   - Add optional filters for display and analysis, but never overwrite raw coordinates.
   - Suggested filters: none, exponential moving average, One Euro filter, Kalman filter.
   - CSV design: `x_raw`, `y_raw`, `z_raw`, `x_filtered`, `y_filtered`, `z_filtered`, `filter_name`, `filter_params`.

### P1 - Live Lab Reliability

6. Improve camera diagnostics.
   - `pose_jitter.py cameras` should print backend name, actual width/height/FPS, FOURCC, exposure, autofocus, and buffer size when available.
   - Add warnings when the actual camera settings differ from requested settings.

7. Save video sidecars with CSV.
   - Record an MP4/AVI sidecar so high-jitter frames can be visually audited.
   - Add frame indexes on the video overlay to match CSV rows.

8. Add dashboard health metrics.
   - Show pose-lock rate, invalid landmark count, actual FPS, dropped-frame count, shoulder-width coefficient of variation, and top invalid joints.

9. Add optional MediaPipe `LIVE_STREAM` mode.
   - Keep the current `VIDEO` mode for deterministic frame-by-frame analysis.
   - Add async live mode for responsiveness and explicitly report callback latency and dropped frames.

### P1 - Stereo and Millimeter Validity

10. Build a stereo calibration wizard.
    - Capture ChArUco or checkerboard images for each camera.
    - Run individual camera calibration, stereo calibration, stereo rectification, and calibration quality reporting.
    - Store intrinsics, distortion coefficients, rotation, translation, projection matrices, reprojection error, and square size.

11. Use synchronized stereo capture behavior.
    - For two webcams, call `grab()` on both cameras first, then `retrieve()` each frame.
    - This reduces temporal skew compared with reading left and right sequentially.

12. Only claim millimeters after physical calibration.
    - Stereo output becomes millimeter-scaled only when the calibration target square size and stereo geometry are known.
    - Until then, stereo mode should remain a preview or use arbitrary units.

13. Add calibrated triangulation.
    - Rectify frames with saved calibration maps.
    - Match corresponding landmarks across left/right views.
    - Use projection matrices with `triangulatePoints`.
    - Store 3D points, reprojection error, epipolar error, and invalid-reconstruction reasons.

### P2 - Thesis Analysis Strength

14. Add validation-study exports.
    - Produce tables for repeated trials, baseline trials, matched MediaPipe/stereo trials, RMSE, correlation, Bland-Altman differences, and equivalence bounds.

15. Expand the HTML report.
    - Add invalid-frame rate, actual FPS distribution, frame-gap histogram, velocity summaries, confidence distributions, model provenance, camera settings, and calibration quality.

16. Add synthetic stereo tests.
    - Generate known 3D points, project them into two virtual cameras, triangulate them, and confirm reconstruction error stays below a tight threshold.

17. Add CI and reproducible environments.
    - Add GitHub Actions or a local `tox`/`nox` style runner.
    - Pin live dependencies in a `requirements-live.txt` or lock file for the student machine.

## Methodological Guardrails

- MediaPipe normalized x/y coordinates should be treated as image-normalized values, not millimeters.
- MediaPipe z in normalized landmarks should not be treated as calibrated physical depth.
- MediaPipe world landmarks can be useful, but they are model-estimated 3D values and should be validated separately before being used as ground truth.
- A stereo rig needs camera intrinsics, distortion coefficients, stereo extrinsics, rectification, and known physical scale before reporting millimeters.
- Filtering is acceptable for display and secondary analysis, but the thesis should report exactly which metrics use raw coordinates and which use filtered coordinates.
- Per-frame jitter and per-second jitter answer different questions. Use per-frame jitter for frame-to-frame stability and per-second velocity when frame timing is uneven.

## Source Notes

- MediaPipe Pose Landmarker Python documentation: normalized landmarks, world landmarks, running modes, timestamps, and confidence options.  
  https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker/python

- MediaPipe Pose Landmarker overview: model variants, 33 pose landmarks, and pose detection/tracking confidence options.  
  https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker

- OpenCV camera calibration tutorial: intrinsics, distortion, known square size, and reprojection error.  
  https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html

- OpenCV calib3d reference: stereo rectification, projection matrices, and triangulation.  
  https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html

- OpenCV VideoCapture reference: camera properties and `grab()`/`retrieve()` behavior for multi-camera capture.  
  https://docs.opencv.org/4.x/d8/dfe/classcv_1_1VideoCapture.html

- OpenCV ArUco/ChArUco calibration tutorial: ChArUco calibration workflow and accuracy advantages for board corners.  
  https://docs.opencv.org/4.x/da/d13/tutorial_aruco_calibration.html

- BlazePose paper: real-time 33-keypoint pose inference background.  
  https://arxiv.org/abs/2006.10204

- BlazePose GHUM Holistic paper: monocular 3D landmark estimation background.  
  https://arxiv.org/abs/2206.11678

- One Euro Filter: practical jitter-vs-lag filtering approach for noisy interactive signals.  
  https://gery.casiez.net/1euro/

- JMIR Formative Research MediaPipe comparison study: example of normalization, resampling, RMSE, and caution around validation.  
  https://formative.jmir.org/2024/1/e56682

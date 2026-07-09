# Live Pose Jitter Lab

The live lab is the part Zechariah's friend actually needs while developing the thesis: it opens the webcam, draws the body landmarks in real time, shows shoulder-normalized jitter while the subject moves, and records the same long-form CSV format used by the offline analyzer.

## Setup on This Windows Machine

MediaPipe is installed in the Python 3.12 project environment because the default Python 3.13 environment does not expose the required MediaPipe runtime here.

```powershell
py -3.12 -m venv .venv312
.\.venv312\Scripts\python.exe -m pip install --upgrade pip
.\.venv312\Scripts\python.exe -m pip install -e ".[dev,live]"
```

List webcams:

```powershell
.\.venv312\Scripts\python.exe pose_jitter.py cameras
```

Start the live viewer:

```powershell
.\.venv312\Scripts\python.exe pose_jitter.py live --camera 0
```

Start recording immediately:

```powershell
.\.venv312\Scripts\python.exe pose_jitter.py live --camera 0 --auto-record --trial reach_trial_01 --record data\reach_trial_01_live.csv
```

The first run downloads `models\pose_landmarker_lite.task`. After that, the live viewer can start without downloading the model again.

## Window Controls

| Key | Action |
| --- | --- |
| `q` or `Esc` | Quit |
| `r` | Toggle CSV recording |
| `c` | Reset live jitter baseline |
| `l` | Toggle joint labels |
| `s` | Save screenshot to `reports\live` |
| `p` | Pause |
| `h` | Show/hide help |

## What the Viewer Shows

- All 33 MediaPipe pose landmarks.
- A connected pose skeleton.
- Labels on the main joints: shoulders, elbows, wrists, hips, knees, ankles, and nose.
- Real-time FPS.
- Shoulder width in normalized MediaPipe coordinates.
- Rolling shoulder-normalized jitter by joint.
- Valid and rejected landmark counts after visibility/presence checks.
- Recording status and row count.

The live normalized jitter uses:

```text
frame-to-frame Euclidean landmark displacement / rolling median shoulder width
```

By default, the live MediaPipe domain uses normalized 2D image coordinates. This matches the thesis method for comparing MediaPipe-space jitter against stereo/OpenCV-space jitter after each is normalized by shoulder width in its own domain.

Landmarks below `--visibility-min` or `--presence-min` are recorded with `landmark_valid=False` and an `invalid_reason`. They are not used to create a frame-to-frame jitter jump, which prevents brief tracking dropouts from becoming false motion.

## Recorded CSV

The live CSV contains one row per landmark per frame:

```text
source,domain,trial,frame,time_s,joint,x,y,z,visibility,presence,x_pixel,y_pixel,
raw_jitter,normalized_jitter,rolling_mean_jitter,rolling_std_jitter,
delta_time_s,raw_velocity,normalized_velocity,shoulder_width,shoulder_scale,
scale_valid,scale_mode,landmark_valid,invalid_reason
```

It can be analyzed immediately:

```powershell
python pose_jitter.py analyze data\reach_trial_01_live.csv --out reports\reach_trial_01_live
```

If using the Python 3.12 environment:

```powershell
.\.venv312\Scripts\python.exe pose_jitter.py analyze data\reach_trial_01_live.csv --out reports\reach_trial_01_live
```

## Stereo Webcam Preview

For two webcams:

```powershell
.\.venv312\Scripts\python.exe pose_jitter.py stereo-preview --left-camera 0 --right-camera 1
```

This mode previews two pose streams side by side. It does not claim millimeter output by itself. To get real OpenCV millimeter coordinates, the two cameras need a stereo calibration file from a checkerboard/Charuco calibration workflow, including intrinsics, distortion coefficients, rotation, and translation. Without that calibration, there is no defensible way to convert two webcam views into thesis-grade millimeter coordinates.

## Research Notes

- Keep the shoulders visible because they define the live normalization scale.
- Use fixed camera placement, fixed lighting, and a fixed subject distance when recording trials.
- Record a still/no-motion baseline first; this gives a camera-noise/jitter floor.
- Use the same trial naming across MediaPipe and stereo/OpenCV outputs.
- Do not compare raw normalized MediaPipe distances to raw stereo millimeters. Compare shoulder-normalized jitter metrics.

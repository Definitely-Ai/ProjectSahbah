# Shoulder-Normalized Pose Jitter Methodology

## Measurement Definition

For a source system `s`, trial `t`, landmark `j`, and frame `f`, let the landmark coordinate vector be:

```text
p(s,t,j,f) = [x, y]          for normalized 2D MediaPipe data
p(s,t,j,f) = [x, y, z]       for stereo/OpenCV millimeter data
```

Raw frame-to-frame jitter is the Euclidean displacement between consecutive frames:

```text
d(s,t,j,f) = || p(s,t,j,f) - p(s,t,j,f-1) ||
```

Shoulder width in the same coordinate domain is:

```text
w(s,t,f) = || p(s,t,right_shoulder,f) - p(s,t,left_shoulder,f) ||
```

The primary normalized jitter metric is:

```text
J(s,t,j,f) = d(s,t,j,f) / median_f(w(s,t,f))
```

The result is dimensionless and can be read as "shoulder widths per frame."

When timestamps are available, the analyzer also computes a time-normalized form:

```text
V(s,t,j,f) = J(s,t,j,f) / delta_time_seconds
```

This can be read as "shoulder widths per second." Use the per-frame metric to describe frame-to-frame stability and the per-second metric when camera timing, dropped frames, or different FPS settings could affect interpretation.

## Why Not Convert MediaPipe to Millimeters Directly

MediaPipe normalized coordinates are expressed relative to image dimensions, not directly to physical space. A single constant cannot convert those values into millimeters across subjects, camera distances, zoom levels, and body positions. Shoulder normalization avoids a false conversion by scaling each domain internally before comparing statistics across domains.

## Recommended Primary Analysis

Use trial-median shoulder width as the primary denominator:

```text
J_trial_median = raw_landmark_displacement / median_trial_shoulder_width
```

This controls for body size and camera scale while reducing the chance that shoulder landmark jitter contaminates every normalized value.

Report per source, trial, and joint:

- Mean normalized jitter.
- Median normalized jitter.
- Standard deviation and variance of normalized jitter.
- 95th percentile normalized jitter.
- Maximum normalized jitter.
- Raw jitter statistics in native units.
- Raw and normalized velocity statistics when timestamps are available.

## Sensitivity Analysis

Run the same analysis with alternate shoulder modes:

```powershell
python pose_jitter.py analyze data/example_pose.csv --out reports/trial_median --shoulder-mode trial_median
python pose_jitter.py analyze data/example_pose.csv --out reports/pair --shoulder-mode pair
python pose_jitter.py analyze data/example_pose.csv --out reports/frame --shoulder-mode frame
```

Use this to answer whether conclusions depend on the denominator choice. If trial-median and pair-average conclusions agree, the comparison is stronger.

## Quality Checks Before Interpreting Results

Check these before making thesis claims:

- Missing shoulders: normalized jitter is invalid when shoulder width cannot be computed.
- Shoulder spikes: large frame-level shoulder-width jumps usually indicate landmark failure.
- Dropped frames: frame-to-frame jitter assumes consecutive frame ordering.
- Variable frame timing: use `delta_time_s`, `frame_gap`, and normalized velocity to check whether a spike is motion, camera timing, or dropped frames.
- Mixed dimensions: do not combine 2D and 3D coordinates inside one source/domain label.
- Subject differences: compare within matched trials when possible, then aggregate.
- Outliers: inspect high 95th percentile or maximum jitter frames visually.

## Suggested Methods Text

Jitter was computed as the Euclidean displacement of each landmark between consecutive frames within its native coordinate domain. MediaPipe landmarks were evaluated in normalized 2D image coordinates, while stereo/OpenCV landmarks were evaluated in reconstructed millimeter coordinates. To compare jitter across these incompatible coordinate systems, each frame-to-frame displacement was divided by shoulder width measured in the same domain. The primary analysis used the median shoulder width within each trial as the denominator, producing a dimensionless jitter value expressed in shoulder widths per frame. This normalization avoids assuming a fixed conversion between image-normalized coordinates and millimeters while preserving raw native-unit jitter for within-system interpretation.

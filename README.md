# ProjectSahbah — Pose Jitter Lab

**Thesis-grade cross-domain pose jitter analysis toolkit.**

Compare jitter between MediaPipe (normalized 2D) and stereo vision (mm 3D) systems using shoulder-width normalization — a dimensionless metric that eliminates false pixel-to-millimeter comparisons.

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/import/project?template=https://github.com/Definitely-Ai/ProjectSahbah)

## 🌐 Web App (Vercel)

The web app runs entirely in your browser — **your data never leaves your device**.

**→ [Launch Web App](https://projectsahbah.vercel.app)**

- Upload your pose CSV
- Get instant jitter analysis with charts
- Export summary and frame-level CSVs
- Works on mobile, tablet, and desktop
- Dark mode + print-friendly

## 🧪 Python CLI (Advanced)

For batch analysis, live capture, statistical tests, and publication-ready HTML reports.

### Install

```bash
pip install -e .
```

### Quick Start

```bash
# Generate demo data and analyze
python pose_jitter.py demo

# Analyze your own CSV
python pose_jitter.py analyze your_pose_data.csv --out reports/my_analysis

# Batch analyze multiple trials
python pose_jitter.py batch trial_1.csv trial_2.csv trial_3.csv --out reports/batch --export-figures --phase-split
```

### Commands

| Command | Description |
|---------|-------------|
| `analyze` | Analyze a single pose CSV |
| `batch` | Batch analyze multiple CSVs |
| `demo` | Generate demo data and run analysis |
| `live` | Real-time MediaPipe pose viewer |
| `cameras` | List available webcam indexes |
| `stereo-preview` | Dual-camera stereo preview |

### Key Flags

| Flag | Description |
|------|-------------|
| `--shoulder-mode` | `trial_median` (default), `frame`, or `pair` |
| `--export-figures` | Export individual PNG figures |
| `--no-stats` | Skip statistical analysis |
| `--phase-split` | Split trials into movement phases |
| `--n-phases` | Number of phases (default: 3) |

## 📊 What's in the Report

### Statistical Tests
- **Bland-Altman** agreement analysis (bias, limits of agreement, CIs)
- **Mann-Whitney U** for non-parametric median comparison
- **Levene's test** for variance equality
- **Shapiro-Wilk** normality test with Bonferroni correction
- **Cohen's d** effect size
- **ICC** (intraclass correlation coefficient)
- **Bootstrap CIs** (10,000 resamples, no normality assumption)

### Visualizations (9 chart types)
- Heatmap (joint × source)
- Ranked bar chart
- Box plot distribution
- Temporal jitter trace with rolling mean
- Bland-Altman agreement plot
- CDF overlay
- Radar/spider chart
- Cross-domain correlation scatter with R²
- Shoulder width stability with rolling CV

### Data Quality
- Modified Z-Score outlier detection
- Frame gap analysis
- Shoulder landmark reliability scoring
- Per-trial quality scorecard (A/B/C/D grades)
- Convergence analysis (minimum trial length estimation)

## 🔬 Methodology

```
Jitter = frame-to-frame Euclidean displacement / same-domain shoulder width
```

**Thesis wording:** Jitter was computed as the Euclidean displacement of each landmark between consecutive frames within its native coordinate domain. To compare jitter between MediaPipe normalized 2D coordinates and stereo-reconstructed 3D millimeter coordinates, each displacement was divided by shoulder width measured in the same coordinate domain.

## 📁 CSV Format

Long-form CSV with one landmark per row:

```csv
source,domain,trial,frame,joint,x,y,z,x_mm,y_mm,z_mm
mediapipe_lab,normalized_2d,reach_trial_01,0,left_wrist,0.42,0.65,0.01,,,
stereo_rig,mm_3d,reach_trial_01,0,left_wrist,,,,152.3,287.1,45.6
```

Column aliases are auto-detected (e.g., `X_Norm`, `x_normalized`, `landmark`, `keypoint`).

## 🧪 Tests

```bash
python -m pytest -v
# 60 passed
```

## 📂 Project Structure

```
├── web/                    # Vercel web app (static)
│   ├── index.html
│   ├── css/app.css
│   └── js/
│       ├── analysis.js     # Client-side jitter analysis
│       ├── charts.js       # Chart.js visualizations
│       ├── logger.js       # Structured logging
│       └── app.js          # App controller
├── src/pose_jitter_lab/    # Python package
│   ├── metrics.py          # Core jitter math
│   ├── stats.py            # Statistical tests (Bland-Altman, ICC, etc.)
│   ├── quality.py          # Outlier detection, quality scorecard
│   ├── aggregate.py        # Multi-trial batch analysis
│   ├── report.py           # HTML report generator
│   ├── io.py               # CSV I/O with alias detection
│   ├── live.py             # Real-time MediaPipe viewer
│   └── cli.py              # CLI entry point
├── tests/                  # Pytest test suite
├── vercel.json             # Vercel deployment config
└── pyproject.toml          # Python project config
```

## License

MIT

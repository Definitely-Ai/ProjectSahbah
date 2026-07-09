"""Data quality assessment and outlier detection for pose jitter analysis.

Provides automatic outlier flagging, frame gap analysis, shoulder reliability
scoring, convergence estimation, and a per-trial quality scorecard.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Outlier detection
# ---------------------------------------------------------------------------

def modified_z_scores(values: np.ndarray) -> np.ndarray:
    """Compute Modified Z-Scores using median absolute deviation (MAD).

    More robust to non-normal distributions than standard Z-scores.
    A |Modified Z-Score| > 3.5 is typically flagged as an outlier (Iglewicz & Hoaglin).
    """
    x = np.asarray(values, dtype=float)
    median = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - median))
    if mad < 1e-15:
        return np.zeros_like(x)
    return 0.6745 * (x - median) / mad


def flag_outliers(
    jitter: pd.DataFrame,
    column: str = "normalized_jitter",
    threshold: float = 3.5,
) -> pd.DataFrame:
    """Flag outlier rows using Modified Z-Score on the specified column.

    Returns a copy of the DataFrame with added columns:
    - `mod_z_score`: The Modified Z-Score for each row.
    - `is_outlier`: Boolean flag.
    - `outlier_method`: Description of the method used.
    """
    df = jitter.copy()
    if column not in df.columns or df[column].dropna().empty:
        df["mod_z_score"] = np.nan
        df["is_outlier"] = False
        df["outlier_method"] = ""
        return df

    groups = ["source", "domain", "trial", "joint"]
    df["mod_z_score"] = np.nan
    for _, group in df.groupby(groups, sort=False):
        vals = group[column].values
        z = modified_z_scores(vals)
        df.loc[group.index, "mod_z_score"] = z

    df["is_outlier"] = df["mod_z_score"].abs() > threshold
    df["outlier_method"] = f"Modified Z-Score > {threshold}"
    return df


def iqr_fence(
    values: np.ndarray,
    factor: float = 1.5,
) -> tuple[float, float]:
    """Compute IQR-based fences for outlier detection."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 4:
        return (np.nan, np.nan)
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    return (q1 - factor * iqr, q3 + factor * iqr)


# ---------------------------------------------------------------------------
# Frame gap analysis
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FrameGapReport:
    """Summary of frame continuity within a trial."""
    source: str
    trial: str
    total_frames: int
    expected_frames: int
    missing_frames: int
    max_gap: int
    gap_locations: list[tuple[int, int, int]]  # (frame_from, frame_to, gap_size)
    continuity_pct: float


def frame_gap_analysis(jitter: pd.DataFrame) -> list[FrameGapReport]:
    """Detect dropped or skipped frames that inflate jitter values.

    Frame-to-frame jitter assumes consecutive frames. Any gap > 1 frame means
    the jitter calculation spans more real-world time than a single frame step.
    """
    if jitter.empty or "frame_gap" not in jitter.columns:
        return []

    reports: list[FrameGapReport] = []
    for (source, trial), group in jitter.groupby(["source", "trial"], sort=True):
        gaps = group[group["frame_gap"] > 1]
        gap_locations = [
            (int(row["frame_from"]), int(row["frame_to"]), int(row["frame_gap"]))
            for _, row in gaps.iterrows()
        ]
        total = len(group) + 1  # +1 because jitter rows = frame_count - 1
        frame_min = int(group["frame_from"].min())
        frame_max = int(group["frame_to"].max())
        expected = frame_max - frame_min + 1
        missing = expected - total
        max_gap = int(group["frame_gap"].max()) if not group.empty else 1
        continuity = (1 - len(gaps) / len(group)) * 100 if len(group) > 0 else 100.0

        reports.append(FrameGapReport(
            source=str(source), trial=str(trial),
            total_frames=total, expected_frames=expected,
            missing_frames=max(0, missing), max_gap=max_gap,
            gap_locations=gap_locations,
            continuity_pct=continuity,
        ))

    return reports


# ---------------------------------------------------------------------------
# Shoulder reliability scoring
# ---------------------------------------------------------------------------

def shoulder_reliability(
    shoulder_widths: pd.DataFrame,
    spike_threshold_pct: float = 15.0,
    window: int = 15,
) -> pd.DataFrame:
    """Score shoulder landmark reliability per trial.

    Flags frames where shoulder width changes by > spike_threshold_pct from
    the rolling median. Returns per-trial statistics.
    """
    if shoulder_widths.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    for (source, domain, trial), group in shoulder_widths.groupby(
        ["source", "domain", "trial"], sort=True
    ):
        valid = group.dropna(subset=["shoulder_width"]).sort_values("frame")
        if len(valid) < 3:
            continue

        widths = valid["shoulder_width"].values
        rolling_median = pd.Series(widths).rolling(window, center=True, min_periods=3).median().values
        pct_change = np.abs(widths - rolling_median) / rolling_median * 100

        spike_count = int(np.sum(pct_change > spike_threshold_pct))
        mean_width = float(np.mean(widths))
        std_width = float(np.std(widths, ddof=1))
        cv = std_width / mean_width if mean_width > 1e-15 else np.nan
        median_width = float(np.median(widths))

        rows.append({
            "source": source, "domain": domain, "trial": trial,
            "n_frames": len(valid),
            "mean_width": mean_width,
            "median_width": median_width,
            "std_width": std_width,
            "cv": cv,
            "spike_count": spike_count,
            "spike_pct": spike_count / len(valid) * 100,
            "max_pct_change": float(np.nanmax(pct_change)) if len(pct_change) > 0 else np.nan,
            "reliability": _shoulder_reliability_label(cv, spike_count / len(valid)),
        })

    return pd.DataFrame(rows)


def _shoulder_reliability_label(cv: float, spike_fraction: float) -> str:
    if np.isnan(cv):
        return "insufficient_data"
    if cv < 0.03 and spike_fraction < 0.02:
        return "excellent"
    if cv < 0.06 and spike_fraction < 0.05:
        return "good"
    if cv < 0.10 and spike_fraction < 0.10:
        return "acceptable"
    return "review_needed"


# ---------------------------------------------------------------------------
# Per-trial data quality scorecard
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class QualityScore:
    source: str
    trial: str
    completeness_pct: float
    continuity_pct: float
    shoulder_reliability: str
    outlier_pct: float
    n_frames: int
    overall_grade: str
    notes: list[str]


def quality_scorecard(
    jitter: pd.DataFrame,
    shoulder_widths: pd.DataFrame,
) -> list[QualityScore]:
    """Generate a per-trial quality scorecard combining all quality metrics."""
    flagged = flag_outliers(jitter)
    gaps = {(r.source, r.trial): r for r in frame_gap_analysis(jitter)}
    shoulder = shoulder_reliability(shoulder_widths)

    scores: list[QualityScore] = []
    for (source, trial), group in flagged.groupby(["source", "trial"], sort=True):
        notes: list[str] = []

        # Completeness
        valid = group["normalized_jitter"].notna().sum()
        total = len(group)
        completeness = valid / total * 100 if total > 0 else 0.0
        if completeness < 95:
            notes.append(f"Only {completeness:.1f}% of jitter rows have valid normalization.")

        # Continuity
        gap = gaps.get((source, trial))
        continuity = gap.continuity_pct if gap else 100.0
        if gap and gap.max_gap > 3:
            notes.append(f"Maximum frame gap of {gap.max_gap} frames detected.")

        # Shoulder
        s_rows = shoulder[(shoulder["source"] == source) & (shoulder["trial"] == trial)]
        s_label = s_rows.iloc[0]["reliability"] if not s_rows.empty else "unknown"
        if s_label in ("review_needed",):
            notes.append("Shoulder landmark instability detected — review raw video.")

        # Outliers
        outlier_pct = group["is_outlier"].sum() / total * 100 if total > 0 else 0.0
        if outlier_pct > 5:
            notes.append(f"{outlier_pct:.1f}% of jitter values are statistical outliers.")

        # Overall grade
        grade = _overall_grade(completeness, continuity, s_label, outlier_pct)

        scores.append(QualityScore(
            source=str(source), trial=str(trial),
            completeness_pct=completeness, continuity_pct=continuity,
            shoulder_reliability=s_label, outlier_pct=outlier_pct,
            n_frames=total, overall_grade=grade, notes=notes,
        ))

    return scores


def _overall_grade(completeness: float, continuity: float, shoulder: str, outlier_pct: float) -> str:
    if completeness >= 98 and continuity >= 98 and shoulder in ("excellent", "good") and outlier_pct < 3:
        return "A"
    if completeness >= 95 and continuity >= 95 and shoulder != "review_needed" and outlier_pct < 5:
        return "B"
    if completeness >= 90 and continuity >= 90 and outlier_pct < 10:
        return "C"
    return "D"


# ---------------------------------------------------------------------------
# Convergence analysis (minimum viable trial length)
# ---------------------------------------------------------------------------

def convergence_analysis(
    jitter: pd.DataFrame,
    min_fraction: float = 0.1,
    steps: int = 20,
    n_bootstrap: int = 500,
    seed: int = 42,
) -> pd.DataFrame:
    """Estimate how many frames are needed for stable jitter statistics.

    Bootstraps the mean normalized jitter from progressively larger subsets
    of the trial and computes the CI width. When the CI width stabilizes,
    the sample size is sufficient.
    """
    if jitter.empty:
        return pd.DataFrame()

    rng = np.random.default_rng(seed)
    rows: list[dict] = []

    for (source, trial, joint), group in jitter.groupby(
        ["source", "trial", "joint"], sort=True
    ):
        values = group["normalized_jitter"].dropna().values
        n = len(values)
        if n < 10:
            continue

        fractions = np.linspace(min_fraction, 1.0, steps)
        for frac in fractions:
            sample_size = max(3, int(n * frac))
            boot_means = []
            for _ in range(n_bootstrap):
                sample = rng.choice(values, size=sample_size, replace=True)
                boot_means.append(float(np.mean(sample)))
            boot_means_arr = np.array(boot_means)
            ci_lo, ci_hi = np.percentile(boot_means_arr, [2.5, 97.5])

            rows.append({
                "source": source, "trial": trial, "joint": joint,
                "sample_fraction": float(frac),
                "sample_size": sample_size,
                "boot_mean": float(np.mean(boot_means_arr)),
                "ci_lower": float(ci_lo),
                "ci_upper": float(ci_hi),
                "ci_width": float(ci_hi - ci_lo),
            })

    return pd.DataFrame(rows)

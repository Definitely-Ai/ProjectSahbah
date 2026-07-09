"""Multi-trial aggregation, batch analysis, and movement phase segmentation.

This module supports the thesis workflow where multiple CSV files (e.g., one
per trial, or one per subject) need to be combined into a single cross-trial
analysis with proper random-effects-style aggregation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from .io import load_pose_csv, write_csvs
from .metrics import compute_jitter, summarize_jitter, compare_sources, ShoulderMode
from .stats import (
    bland_altman_from_jitter,
    bootstrap_ci,
    cohens_d,
    cohens_d_label,
    compute_icc,
    cross_domain_tests,
    joint_reliability,
    normality_tests,
    run_full_stats,
)
from .quality import quality_scorecard, shoulder_reliability


# ---------------------------------------------------------------------------
# Batch loading
# ---------------------------------------------------------------------------

def load_batch(
    paths: list[Path],
    *,
    source_from_filename: bool = False,
    trial_from_filename: bool = True,
) -> pd.DataFrame:
    """Load and concatenate multiple pose CSVs into one DataFrame.

    Parameters
    ----------
    paths : list[Path]
        CSV file paths to load.
    source_from_filename : bool
        If True, override the ``source`` column with the filename stem.
    trial_from_filename : bool
        If True, override the ``trial`` column with the filename stem
        when the original CSV has no ``trial`` column or all rows share
        the default ``trial_1`` value.
    """
    frames: list[pd.DataFrame] = []
    for path in paths:
        df = load_pose_csv(path)
        stem = Path(path).stem

        if source_from_filename:
            df["source"] = stem

        if trial_from_filename:
            # Only override if the trial column was auto-assigned
            if df["trial"].nunique() == 1 and df["trial"].iloc[0] in ("trial_1", stem):
                df["trial"] = stem

        df["_source_file"] = str(path)
        frames.append(df)

    if not frames:
        raise ValueError("No CSV files were provided for batch loading.")

    combined = pd.concat(frames, ignore_index=True)
    sort_cols = [c for c in ["source", "domain", "trial", "frame", "joint"] if c in combined.columns]
    return combined.sort_values(sort_cols).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Cross-trial consistency (ICC across repeated trials)
# ---------------------------------------------------------------------------

def cross_trial_consistency(summary: pd.DataFrame) -> pd.DataFrame:
    """Compute per-joint ICC across trials within each source.

    This answers: "How repeatable are jitter measurements across trials?"
    High ICC means the system consistently produces similar jitter profiles
    across different recording sessions.
    """
    if summary.empty or summary["trial"].nunique() < 2:
        return pd.DataFrame()

    rows: list[dict] = []
    for (source, domain, joint), group in summary.groupby(
        ["source", "domain", "joint"], sort=True
    ):
        trials = group.sort_values("trial")
        values = trials["normalized_mean"].dropna().values
        if len(values) < 3:
            continue

        # Use split-half approach: compare first half vs second half
        mid = len(values) // 2
        if mid < 2:
            continue
        a = values[:mid]
        b = values[mid:2 * mid]
        icc_result = compute_icc(a, b)
        pt, ci_lo, ci_hi = bootstrap_ci(values, statistic="mean", n_bootstrap=5000)

        rows.append({
            "source": source,
            "domain": domain,
            "joint": joint,
            "n_trials": len(values),
            "mean_across_trials": float(np.mean(values)),
            "std_across_trials": float(np.std(values, ddof=1)),
            "cv_across_trials": float(np.std(values, ddof=1) / np.mean(values))
                if np.mean(values) > 1e-15 else np.nan,
            "ci_lower_95": ci_lo,
            "ci_upper_95": ci_hi,
            "consistency_note": _consistency_note(float(np.std(values, ddof=1) / np.mean(values))
                if np.mean(values) > 1e-15 else np.nan),
        })

    return pd.DataFrame(rows)


def _consistency_note(cv: float) -> str:
    if np.isnan(cv):
        return "Insufficient data"
    if cv < 0.1:
        return "Highly consistent across trials"
    if cv < 0.25:
        return "Moderately consistent"
    return "Variable across trials — investigate trial conditions"


# ---------------------------------------------------------------------------
# Movement phase segmentation
# ---------------------------------------------------------------------------

PhaseMethod = Literal["velocity_threshold", "equal_split", "custom_boundaries"]


def segment_phases(
    jitter: pd.DataFrame,
    method: PhaseMethod = "equal_split",
    n_phases: int = 3,
    phase_names: list[str] | None = None,
    velocity_column: str = "raw_velocity",
    velocity_threshold: float | None = None,
) -> pd.DataFrame:
    """Split trials into movement phases and label each jitter row.

    Parameters
    ----------
    method : str
        - ``equal_split``: Divide each trial into ``n_phases`` equal-length segments.
        - ``velocity_threshold``: Split into "moving" and "still" phases based on
          velocity exceeding a threshold.
        - ``custom_boundaries``: Use ``phase_names`` and ``n_phases`` to split
          at evenly-spaced frame boundaries with custom labels.
    n_phases : int
        Number of phases for equal_split / custom_boundaries.
    phase_names : list[str] or None
        Custom labels for each phase. Defaults to "phase_1", "phase_2", etc.
    velocity_threshold : float or None
        For velocity_threshold method. If None, uses median raw velocity.
    """
    if jitter.empty:
        return jitter.copy()

    df = jitter.copy()
    names = phase_names or [f"phase_{i + 1}" for i in range(n_phases)]

    if method == "velocity_threshold":
        if velocity_column not in df.columns:
            df["phase"] = "unknown"
            return df
        if velocity_threshold is None:
            velocity_threshold = float(df[velocity_column].dropna().median())
        df["phase"] = df[velocity_column].apply(
            lambda v: "moving" if pd.notna(v) and v > velocity_threshold else "still"
        )

    elif method == "equal_split":
        for (source, trial), group in df.groupby(["source", "trial"], sort=True):
            mask = (df["source"] == source) & (df["trial"] == trial)
            frames = group["frame_to"].values
            if len(frames) == 0:
                continue
            boundaries = np.linspace(frames.min(), frames.max() + 1, n_phases + 1)
            labels = []
            for frame in frames:
                for i in range(n_phases):
                    if boundaries[i] <= frame < boundaries[i + 1]:
                        labels.append(names[i] if i < len(names) else f"phase_{i + 1}")
                        break
                else:
                    labels.append(names[-1] if names else f"phase_{n_phases}")
            df.loc[mask, "phase"] = labels

    elif method == "custom_boundaries":
        for (source, trial), group in df.groupby(["source", "trial"], sort=True):
            mask = (df["source"] == source) & (df["trial"] == trial)
            frames = group["frame_to"].values
            if len(frames) == 0:
                continue
            boundaries = np.linspace(frames.min(), frames.max() + 1, n_phases + 1)
            labels = []
            for frame in frames:
                for i in range(n_phases):
                    if boundaries[i] <= frame < boundaries[i + 1]:
                        labels.append(names[i] if i < len(names) else f"phase_{i + 1}")
                        break
                else:
                    labels.append(names[-1] if names else f"phase_{n_phases}")
            df.loc[mask, "phase"] = labels
    else:
        raise ValueError(f"Unknown segmentation method: {method}")

    if "phase" not in df.columns:
        df["phase"] = "phase_1"

    return df


def summarize_by_phase(jitter_with_phases: pd.DataFrame) -> pd.DataFrame:
    """Summarize normalized jitter per source/trial/joint/phase."""
    if jitter_with_phases.empty or "phase" not in jitter_with_phases.columns:
        return pd.DataFrame()

    group_cols = ["source", "domain", "trial", "joint", "phase"]
    available = [c for c in group_cols if c in jitter_with_phases.columns]
    grouped = jitter_with_phases.groupby(available, sort=True)

    rows: list[dict] = []
    for key, group in grouped:
        vals = group["normalized_jitter"].dropna()
        if len(vals) < 1:
            continue
        record = dict(zip(available, key if isinstance(key, tuple) else (key,)))
        record.update({
            "n_frames": len(vals),
            "mean": float(vals.mean()),
            "median": float(vals.median()),
            "std": float(vals.std()),
            "p95": float(np.percentile(vals, 95)) if len(vals) >= 2 else np.nan,
            "max": float(vals.max()),
        })
        rows.append(record)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Batch analysis pipeline
# ---------------------------------------------------------------------------

def run_batch(
    paths: list[Path],
    *,
    output_dir: Path,
    shoulder_mode: ShoulderMode = "trial_median",
    left_shoulder: str = "left_shoulder",
    right_shoulder: str = "right_shoulder",
    source_from_filename: bool = False,
    trial_from_filename: bool = True,
    phase_split: bool = False,
    n_phases: int = 3,
    phase_names: list[str] | None = None,
    export_figures: bool = False,
    include_stats: bool = True,
    title: str = "Batch Pose Jitter Analysis",
) -> dict[str, Path]:
    """Run a complete batch analysis on multiple CSV files.

    This is the top-level function for the ``batch`` CLI command.
    """
    from .report import generate_report

    # Load and combine
    pose = load_batch(
        paths,
        source_from_filename=source_from_filename,
        trial_from_filename=trial_from_filename,
    )

    # Compute jitter
    jitter, shoulder_widths = compute_jitter(
        pose,
        shoulder_mode=shoulder_mode,
        left_shoulder=left_shoulder,
        right_shoulder=right_shoulder,
    )

    # Summarize
    summary = summarize_jitter(jitter)
    comparison = compare_sources(summary)

    # Phase segmentation
    phase_summary = pd.DataFrame()
    if phase_split and not jitter.empty:
        jitter_phased = segment_phases(jitter, method="equal_split",
                                       n_phases=n_phases, phase_names=phase_names)
        phase_summary = summarize_by_phase(jitter_phased)

    # Cross-trial consistency
    consistency = cross_trial_consistency(summary)

    # Statistical analysis
    stats_output: dict[str, pd.DataFrame] = {}
    if include_stats and not jitter.empty:
        stats_output = run_full_stats(jitter, summary)

    # Write CSVs
    tables: dict[str, pd.DataFrame] = {
        "jitter_frames": jitter,
        "jitter_summary": summary,
        "comparison": comparison,
        "shoulder_widths": shoulder_widths,
    }
    if not phase_summary.empty:
        tables["phase_summary"] = phase_summary
    if not consistency.empty:
        tables["cross_trial_consistency"] = consistency
    for name, table in stats_output.items():
        tables[f"stats_{name}"] = table

    written = write_csvs(tables, output_dir)

    # Generate report
    report_path = generate_report(
        pose=pose,
        jitter=jitter,
        summary=summary,
        shoulder_widths=shoulder_widths,
        output_path=output_dir / "report.html",
        title=title,
        shoulder_mode=shoulder_mode,
    )
    written["report"] = report_path

    # Export figures
    if export_figures:
        _export_figures(jitter, summary, shoulder_widths, output_dir / "figures")
        written["figures_dir"] = output_dir / "figures"

    return written


def _export_figures(
    jitter: pd.DataFrame,
    summary: pd.DataFrame,
    shoulder_widths: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Export individual publication-ready PNG figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    from .report import (
        STYLE_RC,
        plot_joint_heatmap,
        plot_normalized_by_joint,
        plot_distribution,
        plot_temporal_trace,
        plot_shoulder_widths,
        plot_cdf_overlay,
        plot_radar,
        plot_correlation_scatter,
    )
    from .stats import bland_altman_from_jitter
    from .report import plot_bland_altman

    ba_results, _ = bland_altman_from_jitter(jitter)

    chart_fns = {
        "heatmap": lambda: plot_joint_heatmap(summary),
        "jitter_by_joint": lambda: plot_normalized_by_joint(summary),
        "distribution": lambda: plot_distribution(jitter),
        "temporal_trace": lambda: plot_temporal_trace(jitter),
        "shoulder_stability": lambda: plot_shoulder_widths(shoulder_widths),
        "cdf": lambda: plot_cdf_overlay(jitter),
        "radar": lambda: plot_radar(summary),
        "correlation": lambda: plot_correlation_scatter(jitter),
    }

    if ba_results:
        chart_fns["bland_altman"] = lambda: plot_bland_altman(ba_results)

    with plt.rc_context(STYLE_RC):
        for name, fn in chart_fns.items():
            html = fn()
            if html and "base64" in html:
                # Extract the base64 PNG from the HTML img tag
                import base64
                start = html.find("base64,") + 7
                end = html.find('"', start)
                png_data = base64.b64decode(html[start:end])
                path = output_dir / f"{name}.png"
                path.write_bytes(png_data)

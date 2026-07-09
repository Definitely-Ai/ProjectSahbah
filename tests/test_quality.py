"""Tests for the data quality assessment module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pose_jitter_lab.quality import (
    convergence_analysis,
    flag_outliers,
    frame_gap_analysis,
    iqr_fence,
    modified_z_scores,
    quality_scorecard,
    shoulder_reliability,
)


# ── Modified Z-Score ───────────────────────────────────────────────────────

def test_modified_z_scores_detects_outlier() -> None:
    values = np.array([1.0, 1.1, 0.9, 1.0, 1.05, 1.0, 0.95, 1.0, 10.0])
    z = modified_z_scores(values)
    assert z[-1] > 3.5  # The 10.0 should be an outlier
    assert all(abs(z_i) < 3.5 for z_i in z[:-1])


def test_modified_z_scores_constant_values() -> None:
    values = np.array([5.0, 5.0, 5.0, 5.0])
    z = modified_z_scores(values)
    assert np.allclose(z, 0.0)


# ── Flag outliers ──────────────────────────────────────────────────────────

def test_flag_outliers_adds_columns() -> None:
    df = pd.DataFrame({
        "source": "mp", "domain": "2d", "trial": "t1", "joint": "wrist",
        "normalized_jitter": [0.05, 0.04, 0.06, 0.05, 0.04, 0.05, 0.06, 1.5],
    })
    result = flag_outliers(df)
    assert "is_outlier" in result.columns
    assert "mod_z_score" in result.columns
    assert result.iloc[-1]["is_outlier"]  # 1.5 should be flagged
    assert not result.iloc[0]["is_outlier"]


def test_flag_outliers_empty_df() -> None:
    df = pd.DataFrame(columns=["source", "domain", "trial", "joint", "normalized_jitter"])
    result = flag_outliers(df)
    assert "is_outlier" in result.columns


# ── IQR fencing ────────────────────────────────────────────────────────────

def test_iqr_fence_basic() -> None:
    values = np.arange(1.0, 101.0)
    lower, upper = iqr_fence(values)
    assert lower < 1.0
    assert upper > 100.0


def test_iqr_fence_too_few() -> None:
    lower, upper = iqr_fence(np.array([1.0, 2.0]))
    assert np.isnan(lower)


# ── Frame gap analysis ────────────────────────────────────────────────────

def test_frame_gap_analysis_detects_gaps() -> None:
    jitter = pd.DataFrame({
        "source": ["mp"] * 5,
        "trial": ["t1"] * 5,
        "frame_from": [0, 1, 2, 3, 10],
        "frame_to": [1, 2, 3, 4, 11],
        "frame_gap": [1, 1, 1, 1, 7],
    })
    reports = frame_gap_analysis(jitter)
    assert len(reports) == 1
    report = reports[0]
    assert report.max_gap == 7
    assert len(report.gap_locations) == 1
    assert report.gap_locations[0][2] == 7  # gap size


def test_frame_gap_analysis_no_gaps() -> None:
    jitter = pd.DataFrame({
        "source": ["mp"] * 4,
        "trial": ["t1"] * 4,
        "frame_from": [0, 1, 2, 3],
        "frame_to": [1, 2, 3, 4],
        "frame_gap": [1, 1, 1, 1],
    })
    reports = frame_gap_analysis(jitter)
    assert reports[0].max_gap == 1
    assert len(reports[0].gap_locations) == 0


# ── Shoulder reliability ──────────────────────────────────────────────────

def test_shoulder_reliability_stable() -> None:
    rng = np.random.default_rng(42)
    shoulder_widths = pd.DataFrame({
        "source": "mp", "domain": "2d", "trial": "t1",
        "frame": range(100),
        "shoulder_width": 0.2 + rng.normal(0, 0.002, 100),
    })
    result = shoulder_reliability(shoulder_widths)
    assert not result.empty
    assert result.iloc[0]["reliability"] in ("excellent", "good")
    assert result.iloc[0]["cv"] < 0.05


def test_shoulder_reliability_unstable() -> None:
    widths = np.concatenate([
        np.full(50, 0.2),
        np.full(50, 0.5),  # Big jump
    ])
    shoulder_widths = pd.DataFrame({
        "source": "mp", "domain": "2d", "trial": "t1",
        "frame": range(100),
        "shoulder_width": widths,
    })
    result = shoulder_reliability(shoulder_widths)
    assert result.iloc[0]["reliability"] in ("acceptable", "review_needed")


# ── Quality scorecard ─────────────────────────────────────────────────────

def test_quality_scorecard_grade_a() -> None:
    rng = np.random.default_rng(42)
    n = 100
    jitter = pd.DataFrame({
        "source": "mp", "domain": "2d", "trial": "t1", "joint": "wrist",
        "frame_from": range(n),
        "frame_to": range(1, n + 1),
        "frame_gap": [1] * n,
        "raw_jitter": rng.normal(0.01, 0.002, n),
        "normalized_jitter": rng.normal(0.05, 0.005, n),
        "shoulder_scale": [0.2] * n,
        "scale_valid": [True] * n,
    })
    shoulder_widths = pd.DataFrame({
        "source": "mp", "domain": "2d", "trial": "t1",
        "frame": range(n),
        "shoulder_width": 0.2 + rng.normal(0, 0.002, n),
    })
    scores = quality_scorecard(jitter, shoulder_widths)
    assert len(scores) == 1
    assert scores[0].overall_grade in ("A", "B")


# ── Convergence analysis ──────────────────────────────────────────────────

def test_convergence_analysis_produces_ci_narrowing() -> None:
    rng = np.random.default_rng(42)
    jitter = pd.DataFrame({
        "source": "mp", "trial": "t1", "joint": "wrist",
        "normalized_jitter": rng.normal(0.05, 0.01, 200),
    })
    result = convergence_analysis(jitter, steps=5, n_bootstrap=100)
    assert not result.empty
    # CI width should generally decrease as sample size increases
    widths = result["ci_width"].values
    assert widths[-1] < widths[0]  # Last (full data) should be tighter than first

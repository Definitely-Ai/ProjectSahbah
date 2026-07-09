"""Tests for the statistical analysis module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pose_jitter_lab.stats import (
    BlandAltmanResult,
    ICCResult,
    autocorrelation_lag1,
    bland_altman,
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


# ── Bland-Altman ───────────────────────────────────────────────────────────

def test_bland_altman_perfect_agreement() -> None:
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = bland_altman(values, values, trial="t", joint="wrist")
    assert result.n == 5
    assert np.isclose(result.mean_diff, 0.0)
    assert np.isclose(result.std_diff, 0.0)
    assert result.percent_within_loa == 100.0


def test_bland_altman_known_bias() -> None:
    a = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    b = a - 5.0
    result = bland_altman(a, b, trial="t", joint="j")
    assert result.n == 5
    assert np.isclose(result.mean_diff, 5.0)
    assert result.percent_within_loa == 100.0


def test_bland_altman_too_few_samples() -> None:
    result = bland_altman(np.array([1.0]), np.array([2.0]))
    assert result.n == 1
    assert np.isnan(result.mean_diff)


def test_bland_altman_from_jitter_with_paired_data() -> None:
    jitter = pd.DataFrame([
        {"source": "mp", "domain": "2d", "trial": "t1", "joint": "wrist", "frame_to": i,
         "normalized_jitter": 0.1 + i * 0.01}
        for i in range(20)
    ] + [
        {"source": "stereo", "domain": "3d", "trial": "t1", "joint": "wrist", "frame_to": i,
         "normalized_jitter": 0.12 + i * 0.01}
        for i in range(20)
    ])
    results, table = bland_altman_from_jitter(jitter)
    assert len(results) == 1
    assert not table.empty
    assert results[0].joint == "wrist"
    assert np.isclose(results[0].mean_diff, -0.02, atol=1e-6)


# ── ICC ────────────────────────────────────────────────────────────────────

def test_icc_perfect_agreement() -> None:
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = compute_icc(a, a)
    assert np.isclose(result.icc, 1.0, atol=1e-6)
    assert result.label == "excellent"


def test_icc_no_agreement() -> None:
    rng = np.random.default_rng(42)
    a = rng.normal(0, 1, 100)
    b = rng.normal(0, 1, 100)
    result = compute_icc(a, b)
    assert result.icc < 0.3
    assert result.label in ("poor", "moderate")


def test_icc_insufficient_data() -> None:
    result = compute_icc(np.array([1.0]), np.array([2.0]))
    assert np.isnan(result.icc)
    assert result.label == "insufficient_data"


# ── Cohen's d ──────────────────────────────────────────────────────────────

def test_cohens_d_identical() -> None:
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert np.isclose(cohens_d(a, a), 0.0)
    assert cohens_d_label(0.0) == "negligible"


def test_cohens_d_large_effect() -> None:
    a = np.array([10.0, 11.0, 12.0, 13.0])
    b = np.array([1.0, 2.0, 3.0, 4.0])
    d = cohens_d(a, b)
    assert d > 0.8
    assert cohens_d_label(d) == "large"


def test_cohens_d_insufficient_data() -> None:
    assert np.isnan(cohens_d(np.array([1.0]), np.array([2.0])))


# ── Autocorrelation ────────────────────────────────────────────────────────

def test_autocorrelation_white_noise() -> None:
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 1, 1000)
    ac = autocorrelation_lag1(noise)
    assert abs(ac) < 0.1


def test_autocorrelation_strong_signal() -> None:
    # Perfect drift: each value is previous + constant
    x = np.cumsum(np.ones(100))
    ac = autocorrelation_lag1(x)
    assert ac > 0.9


def test_autocorrelation_too_few() -> None:
    assert np.isnan(autocorrelation_lag1(np.array([1.0, 2.0])))


# ── Bootstrap CI ───────────────────────────────────────────────────────────

def test_bootstrap_ci_returns_point_and_bounds() -> None:
    values = np.arange(1.0, 101.0)
    point, ci_lo, ci_hi = bootstrap_ci(values, statistic="mean")
    true_mean = 50.5
    assert np.isclose(point, true_mean)
    assert ci_lo < true_mean < ci_hi


def test_bootstrap_ci_too_few() -> None:
    point, lo, hi = bootstrap_ci(np.array([1.0]))
    assert np.isnan(point)
    assert np.isnan(lo)


# ── Normality tests ───────────────────────────────────────────────────────

def test_normality_tests_returns_results() -> None:
    rng = np.random.default_rng(42)
    jitter = pd.DataFrame({
        "source": "mp", "domain": "2d", "trial": "t1", "joint": "wrist",
        "normalized_jitter": rng.normal(0.05, 0.01, 50),
    })
    result = normality_tests(jitter)
    assert not result.empty
    assert "shapiro_stat" in result.columns
    assert "normal" in result.columns


def test_normality_tests_too_few_samples() -> None:
    jitter = pd.DataFrame({
        "source": "mp", "domain": "2d", "trial": "t1", "joint": "wrist",
        "normalized_jitter": [0.1, 0.2],
    })
    result = normality_tests(jitter)
    assert result.iloc[0]["note"] == "Too few samples for normality test (n < 8)."


# ── Cross-domain tests ────────────────────────────────────────────────────

def test_cross_domain_tests_with_two_sources() -> None:
    rng = np.random.default_rng(42)
    n = 50
    jitter = pd.DataFrame(
        [{"source": "mp", "domain": "2d", "trial": "t1", "joint": "wrist",
          "normalized_jitter": 0.05 + rng.normal(0, 0.01)} for _ in range(n)]
        + [{"source": "stereo", "domain": "3d", "trial": "t1", "joint": "wrist",
            "normalized_jitter": 0.08 + rng.normal(0, 0.01)} for _ in range(n)]
    )
    results = cross_domain_tests(jitter)
    assert len(results) >= 2  # Mann-Whitney + Levene
    mann_whitney = [r for r in results if r.test_name == "Mann-Whitney U"]
    assert len(mann_whitney) == 1
    assert mann_whitney[0].significant  # Large difference should be significant


def test_cross_domain_tests_single_source() -> None:
    jitter = pd.DataFrame({
        "source": "mp", "domain": "2d", "trial": "t1", "joint": "wrist",
        "normalized_jitter": [0.1, 0.2, 0.3],
    })
    results = cross_domain_tests(jitter)
    assert len(results) == 0


# ── Joint reliability ──────────────────────────────────────────────────────

def test_joint_reliability_computes_cv_and_ci() -> None:
    rng = np.random.default_rng(42)
    jitter = pd.DataFrame({
        "source": "mp", "domain": "2d", "trial": "t1", "joint": "wrist",
        "normalized_jitter": rng.normal(0.05, 0.01, 50),
    })
    summary = pd.DataFrame({
        "source": "mp", "domain": "2d", "trial": "t1", "joint": "wrist",
        "normalized_mean": [0.05],
    })
    result = joint_reliability(jitter, summary)
    assert not result.empty
    assert "cv" in result.columns
    assert "autocorr_lag1" in result.columns
    assert "ci_lower_95" in result.columns


# ── Full stats runner ──────────────────────────────────────────────────────

def test_run_full_stats_returns_named_tables() -> None:
    rng = np.random.default_rng(42)
    n = 30
    jitter = pd.DataFrame(
        [{"source": "mp", "domain": "2d", "trial": "t1", "joint": "wrist",
          "frame_to": i, "frame_from": i - 1, "frame_gap": 1,
          "raw_jitter": 0.01, "normalized_jitter": 0.05 + rng.normal(0, 0.005),
          "shoulder_scale": 0.2, "scale_valid": True, "shoulder_mode": "trial_median"}
         for i in range(1, n + 1)]
        + [{"source": "stereo", "domain": "3d", "trial": "t1", "joint": "wrist",
            "frame_to": i, "frame_from": i - 1, "frame_gap": 1,
            "raw_jitter": 5.0, "normalized_jitter": 0.08 + rng.normal(0, 0.005),
            "shoulder_scale": 400.0, "scale_valid": True, "shoulder_mode": "trial_median"}
           for i in range(1, n + 1)]
    )
    summary = pd.DataFrame([
        {"source": "mp", "domain": "2d", "trial": "t1", "joint": "wrist", "normalized_mean": 0.05},
        {"source": "stereo", "domain": "3d", "trial": "t1", "joint": "wrist", "normalized_mean": 0.08},
    ])
    output = run_full_stats(jitter, summary)
    assert "normality" in output
    assert "cross_domain_tests" in output
    assert "joint_reliability" in output

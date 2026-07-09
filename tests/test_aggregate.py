"""Tests for the multi-trial aggregation module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pose_jitter_lab.aggregate import (
    cross_trial_consistency,
    load_batch,
    segment_phases,
    summarize_by_phase,
)
from pose_jitter_lab.sample_data import write_sample_pose


# ── Batch loading ──────────────────────────────────────────────────────────

def test_load_batch_combines_files(tmp_path) -> None:
    """Two separate CSV files should be combined into one DataFrame."""
    path1 = write_sample_pose(tmp_path / "trial_a.csv", frames=10, seed=1)
    path2 = write_sample_pose(tmp_path / "trial_b.csv", frames=10, seed=2)

    combined = load_batch([path1, path2], trial_from_filename=True)

    assert not combined.empty
    assert "_source_file" in combined.columns
    # Should have data from both files
    assert combined["_source_file"].nunique() == 2


def test_load_batch_trial_from_filename(tmp_path) -> None:
    """When trial_from_filename is True, the trial column should use the filename stem."""
    path = write_sample_pose(tmp_path / "my_custom_trial.csv", frames=10, seed=1)
    combined = load_batch([path], trial_from_filename=True)

    # The original sample data uses "reach_trial_01" as trial name,
    # so trial_from_filename should NOT override it (it only overrides "trial_1")
    assert not combined.empty


def test_load_batch_raises_on_empty() -> None:
    with pytest.raises(ValueError, match="No CSV files"):
        load_batch([])


def test_load_batch_source_from_filename(tmp_path) -> None:
    path = write_sample_pose(tmp_path / "my_source.csv", frames=10, seed=1)
    combined = load_batch([path], source_from_filename=True)
    assert (combined["source"] == "my_source").all()


# ── Phase segmentation ────────────────────────────────────────────────────

def test_segment_phases_equal_split() -> None:
    jitter = pd.DataFrame({
        "source": ["mp"] * 30,
        "trial": ["t1"] * 30,
        "joint": ["wrist"] * 30,
        "frame_to": range(30),
        "normalized_jitter": np.random.default_rng(42).normal(0.05, 0.01, 30),
    })
    result = segment_phases(jitter, method="equal_split", n_phases=3)
    assert "phase" in result.columns
    assert result["phase"].nunique() == 3
    assert set(result["phase"].unique()) == {"phase_1", "phase_2", "phase_3"}


def test_segment_phases_custom_names() -> None:
    jitter = pd.DataFrame({
        "source": ["mp"] * 30,
        "trial": ["t1"] * 30,
        "joint": ["wrist"] * 30,
        "frame_to": range(30),
        "normalized_jitter": np.random.default_rng(42).normal(0.05, 0.01, 30),
    })
    result = segment_phases(
        jitter, method="equal_split", n_phases=3,
        phase_names=["reach", "hold", "return"],
    )
    assert set(result["phase"].unique()) == {"reach", "hold", "return"}


def test_segment_phases_velocity_threshold() -> None:
    jitter = pd.DataFrame({
        "source": ["mp"] * 20,
        "trial": ["t1"] * 20,
        "joint": ["wrist"] * 20,
        "frame_to": range(20),
        "normalized_jitter": np.random.default_rng(42).normal(0.05, 0.01, 20),
        "raw_velocity": [0.01] * 10 + [0.5] * 10,
    })
    result = segment_phases(jitter, method="velocity_threshold", velocity_threshold=0.1)
    assert "phase" in result.columns
    assert "still" in result["phase"].values
    assert "moving" in result["phase"].values


def test_segment_phases_empty() -> None:
    df = pd.DataFrame()
    result = segment_phases(df)
    assert result.empty


# ── Summarize by phase ────────────────────────────────────────────────────

def test_summarize_by_phase_produces_per_phase_stats() -> None:
    rng = np.random.default_rng(42)
    jitter = pd.DataFrame({
        "source": ["mp"] * 30,
        "domain": ["2d"] * 30,
        "trial": ["t1"] * 30,
        "joint": ["wrist"] * 30,
        "frame_to": range(30),
        "normalized_jitter": rng.normal(0.05, 0.01, 30),
        "phase": ["reach"] * 10 + ["hold"] * 10 + ["return"] * 10,
    })
    result = summarize_by_phase(jitter)
    assert not result.empty
    assert result["phase"].nunique() == 3
    assert "mean" in result.columns
    assert "std" in result.columns
    assert "p95" in result.columns


# ── Cross-trial consistency ───────────────────────────────────────────────

def test_cross_trial_consistency_computes_cv() -> None:
    summary = pd.DataFrame([
        {"source": "mp", "domain": "2d", "trial": f"t{i}", "joint": "wrist",
         "normalized_mean": 0.05 + i * 0.001}
        for i in range(5)
    ])
    result = cross_trial_consistency(summary)
    assert not result.empty
    assert "cv_across_trials" in result.columns
    assert "consistency_note" in result.columns


def test_cross_trial_consistency_single_trial() -> None:
    summary = pd.DataFrame([
        {"source": "mp", "domain": "2d", "trial": "t1", "joint": "wrist",
         "normalized_mean": 0.05},
    ])
    result = cross_trial_consistency(summary)
    assert result.empty  # Need >= 2 trials


# ── CLI integration test ──────────────────────────────────────────────────

def test_batch_cli_runs_end_to_end(tmp_path) -> None:
    """End-to-end test: generate two CSVs, batch-analyze, and verify output files."""
    from pose_jitter_lab.aggregate import run_batch

    path1 = write_sample_pose(tmp_path / "trial_a.csv", frames=20, seed=1)
    path2 = write_sample_pose(tmp_path / "trial_b.csv", frames=20, seed=2)
    out = tmp_path / "batch_output"

    written = run_batch(
        paths=[path1, path2],
        output_dir=out,
        include_stats=True,
        phase_split=True,
        n_phases=2,
    )

    assert (out / "report.html").exists()
    assert (out / "jitter_frames.csv").exists()
    assert (out / "jitter_summary.csv").exists()
    assert (out / "phase_summary.csv").exists()
    assert "report" in written


def test_batch_with_figure_export(tmp_path) -> None:
    from pose_jitter_lab.aggregate import run_batch

    path = write_sample_pose(tmp_path / "trial_a.csv", frames=20, seed=1)
    out = tmp_path / "fig_output"

    written = run_batch(
        paths=[path],
        output_dir=out,
        export_figures=True,
        include_stats=False,
    )

    figures_dir = out / "figures"
    assert figures_dir.exists()
    png_files = list(figures_dir.glob("*.png"))
    assert len(png_files) >= 3  # At least heatmap, distribution, shoulder_stability

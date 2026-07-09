"""Pose jitter normalization tools for thesis-grade movement analysis."""

from .io import load_pose_csv
from .metrics import (
    compute_jitter,
    compute_shoulder_widths,
    compare_sources,
    summarize_jitter,
)
from .stats import (
    bland_altman,
    bland_altman_from_jitter,
    bootstrap_ci,
    cohens_d,
    compute_icc,
    cross_domain_tests,
    joint_reliability,
    normality_tests,
    run_full_stats,
)
from .quality import (
    flag_outliers,
    frame_gap_analysis,
    quality_scorecard,
    shoulder_reliability,
    convergence_analysis,
)
from .aggregate import (
    cross_trial_consistency,
    load_batch,
    run_batch,
    segment_phases,
    summarize_by_phase,
)

__all__ = [
    "bland_altman",
    "bland_altman_from_jitter",
    "bootstrap_ci",
    "cohens_d",
    "compare_sources",
    "compute_icc",
    "compute_jitter",
    "compute_shoulder_widths",
    "convergence_analysis",
    "cross_domain_tests",
    "cross_trial_consistency",
    "flag_outliers",
    "frame_gap_analysis",
    "joint_reliability",
    "load_batch",
    "load_pose_csv",
    "normality_tests",
    "quality_scorecard",
    "run_batch",
    "run_full_stats",
    "segment_phases",
    "shoulder_reliability",
    "summarize_by_phase",
    "summarize_jitter",
]

__version__ = "0.2.0"

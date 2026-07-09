from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Literal

import numpy as np
import pandas as pd

from .io import available_coordinate_columns, coordinate_columns, normalize_label, require_columns

ShoulderMode = Literal["trial_median", "frame", "pair"]


@dataclass(frozen=True)
class ShoulderConfig:
    left: str = "left_shoulder"
    right: str = "right_shoulder"
    mode: ShoulderMode = "trial_median"
    min_width: float = 1e-9

    def normalized(self) -> "ShoulderConfig":
        return ShoulderConfig(
            left=normalize_label(self.left),
            right=normalize_label(self.right),
            mode=self.mode,
            min_width=self.min_width,
        )


def compute_shoulder_widths(
    pose: pd.DataFrame,
    *,
    left_shoulder: str = "left_shoulder",
    right_shoulder: str = "right_shoulder",
) -> pd.DataFrame:
    """Compute left-to-right shoulder width for every frame."""
    require_columns(pose, ["source", "domain", "trial", "frame", "joint", "x", "y"])
    left = normalize_label(left_shoulder)
    right = normalize_label(right_shoulder)

    rows: list[dict[str, object]] = []
    group_columns = ["source", "domain", "trial", "frame"]

    for group_key, group in pose.groupby(group_columns, sort=True):
        left_rows = group[group["joint"] == left]
        right_rows = group[group["joint"] == right]
        record = dict(zip(group_columns, group_key))

        if left_rows.empty or right_rows.empty:
            rows.append({**record, "shoulder_width": np.nan, "shoulder_dims": ""})
            continue

        left_row = left_rows.iloc[0]
        right_row = right_rows.iloc[0]
        dims = [
            column
            for column in coordinate_columns(group)
            if pd.notna(left_row[column]) and pd.notna(right_row[column])
        ]

        if not dims:
            rows.append({**record, "shoulder_width": np.nan, "shoulder_dims": ""})
            continue

        width = euclidean_distance(left_row, right_row, dims)
        rows.append(
            {
                **record,
                "shoulder_width": width,
                "shoulder_dims": "+".join(dims),
            }
        )

    return pd.DataFrame(rows).sort_values(group_columns).reset_index(drop=True)


def compute_jitter(
    pose: pd.DataFrame,
    *,
    shoulder_mode: ShoulderMode = "trial_median",
    left_shoulder: str = "left_shoulder",
    right_shoulder: str = "right_shoulder",
    min_shoulder_width: float = 1e-9,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute frame-to-frame raw and shoulder-normalized landmark jitter.

    Returns `(jitter_frames, shoulder_widths)`.
    """
    require_columns(pose, ["source", "domain", "trial", "frame", "joint", "x", "y"])
    config = ShoulderConfig(
        left=left_shoulder,
        right=right_shoulder,
        mode=shoulder_mode,
        min_width=min_shoulder_width,
    ).normalized()
    if config.mode not in {"trial_median", "frame", "pair"}:
        raise ValueError("shoulder_mode must be one of: trial_median, frame, pair")

    shoulder_widths = compute_shoulder_widths(
        pose,
        left_shoulder=config.left,
        right_shoulder=config.right,
    )
    shoulder_lookup = {
        (row.source, row.domain, row.trial, row.frame): row.shoulder_width
        for row in shoulder_widths.itertuples(index=False)
    }
    trial_scale_lookup = (
        shoulder_widths.groupby(["source", "domain", "trial"])["shoulder_width"]
        .median()
        .to_dict()
    )

    rows: list[dict[str, object]] = []
    group_columns = ["source", "domain", "trial", "joint"]
    sort_columns = ["frame"]
    if "time_s" in pose.columns:
        sort_columns.append("time_s")

    for group_key, group in pose.groupby(group_columns, sort=True):
        source, domain, trial, joint = group_key
        sorted_group = group.sort_values(sort_columns)
        if len(sorted_group) < 2:
            continue

        previous = None
        for current in sorted_group.itertuples(index=False):
            current_series = pd.Series(current._asdict())
            if previous is None:
                previous = current_series
                continue

            dims = common_coordinate_columns(previous, current_series)
            if not dims:
                previous = current_series
                continue

            raw = euclidean_distance(previous, current_series, dims)
            previous_key = (source, domain, trial, previous["frame"])
            current_key = (source, domain, trial, current_series["frame"])
            scale = resolve_shoulder_scale(
                mode=config.mode,
                previous_width=shoulder_lookup.get(previous_key, np.nan),
                current_width=shoulder_lookup.get(current_key, np.nan),
                trial_width=trial_scale_lookup.get((source, domain, trial), np.nan),
            )
            scale_valid = pd.notna(scale) and scale > config.min_width
            normalized = raw / scale if scale_valid else np.nan
            delta_time_s = np.nan
            raw_velocity = np.nan
            normalized_velocity = np.nan
            if "time_s" in previous and "time_s" in current_series:
                delta_time_s = current_series["time_s"] - previous["time_s"]
                if pd.notna(delta_time_s) and delta_time_s > 0:
                    raw_velocity = raw / delta_time_s
                    normalized_velocity = normalized / delta_time_s if scale_valid else np.nan

            row = {
                "source": source,
                "domain": domain,
                "trial": trial,
                "joint": joint,
                "frame_from": previous["frame"],
                "frame_to": current_series["frame"],
                "frame_gap": current_series["frame"] - previous["frame"],
                "raw_jitter": raw,
                "shoulder_scale": scale,
                "normalized_jitter": normalized,
                "delta_time_s": delta_time_s,
                "raw_velocity": raw_velocity,
                "normalized_velocity": normalized_velocity,
                "coordinate_dims": "+".join(dims),
                "shoulder_mode": config.mode,
                "scale_valid": bool(scale_valid),
            }
            if "time_s" in previous and "time_s" in current_series:
                row["time_from_s"] = previous["time_s"]
                row["time_to_s"] = current_series["time_s"]
            rows.append(row)
            previous = current_series

    jitter = pd.DataFrame(rows)
    if jitter.empty:
        return jitter, shoulder_widths
    return jitter.sort_values(["source", "domain", "trial", "joint", "frame_to"]).reset_index(drop=True), shoulder_widths


def summarize_jitter(jitter: pd.DataFrame) -> pd.DataFrame:
    """Summarize normalized and raw jitter per source/domain/trial/joint."""
    if jitter.empty:
        return pd.DataFrame()

    grouped = jitter.groupby(["source", "domain", "trial", "joint"], sort=True)
    aggregations = dict(
        frames=("normalized_jitter", "count"),
        raw_mean=("raw_jitter", "mean"),
        raw_std=("raw_jitter", "std"),
        raw_variance=("raw_jitter", "var"),
        raw_max=("raw_jitter", "max"),
        normalized_mean=("normalized_jitter", "mean"),
        normalized_median=("normalized_jitter", "median"),
        normalized_std=("normalized_jitter", "std"),
        normalized_variance=("normalized_jitter", "var"),
        normalized_p95=("normalized_jitter", percentile_95),
        normalized_max=("normalized_jitter", "max"),
        shoulder_scale=("shoulder_scale", "median"),
    )
    if "raw_velocity" in jitter.columns:
        aggregations.update(
            raw_velocity_mean=("raw_velocity", "mean"),
            raw_velocity_std=("raw_velocity", "std"),
            normalized_velocity_mean=("normalized_velocity", "mean"),
            normalized_velocity_std=("normalized_velocity", "std"),
            normalized_velocity_p95=("normalized_velocity", percentile_95),
        )
    if "frame_gap" in jitter.columns:
        aggregations.update(max_frame_gap=("frame_gap", "max"))

    summary = grouped.agg(**aggregations)
    return summary.reset_index()


def compare_sources(summary: pd.DataFrame) -> pd.DataFrame:
    """Build a side-by-side comparison table for normalized jitter metrics."""
    if summary.empty:
        return pd.DataFrame()

    index_columns = ["trial", "joint"]
    value_columns = [
        "normalized_mean",
        "normalized_median",
        "normalized_std",
        "normalized_variance",
        "normalized_p95",
        "normalized_max",
    ]
    melted = summary.melt(
        id_vars=index_columns + ["source", "domain"],
        value_vars=value_columns,
        var_name="metric",
        value_name="value",
    )
    melted["series"] = melted["source"] + "__" + melted["domain"] + "__" + melted["metric"]
    comparison = (
        melted.pivot_table(index=index_columns, columns="series", values="value", aggfunc="first")
        .reset_index()
        .rename_axis(None, axis=1)
    )
    return comparison


def resolve_shoulder_scale(
    *,
    mode: ShoulderMode,
    previous_width: float,
    current_width: float,
    trial_width: float,
) -> float:
    if mode == "trial_median":
        return trial_width
    if mode == "frame":
        return current_width
    if mode == "pair":
        values = [value for value in [previous_width, current_width] if pd.notna(value)]
        return float(np.mean(values)) if values else np.nan
    raise ValueError(f"Unsupported shoulder mode: {mode}")


def common_coordinate_columns(previous: pd.Series, current: pd.Series) -> list[str]:
    previous_dims = set(available_coordinate_columns(previous))
    current_dims = set(available_coordinate_columns(current))
    return [column for column in ["x", "y", "z"] if column in previous_dims and column in current_dims]


def euclidean_distance(left: pd.Series, right: pd.Series, dims: list[str]) -> float:
    return sqrt(sum((float(left[column]) - float(right[column])) ** 2 for column in dims))


def percentile_95(values: pd.Series) -> float:
    clean = values.dropna()
    if clean.empty:
        return np.nan
    return float(np.percentile(clean, 95))

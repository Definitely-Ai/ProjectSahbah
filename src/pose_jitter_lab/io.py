from __future__ import annotations

from pathlib import Path
from typing import Iterable
import warnings

import pandas as pd


COLUMN_ALIASES = {
    "bodypart": "joint",
    "body_part": "joint",
    "frame_id": "frame",
    "frame_index": "frame",
    "frame_number": "frame",
    "keypoint": "joint",
    "landmark": "joint",
    "landmark_name": "joint",
    "method": "source",
    "node": "joint",
    "pipeline": "source",
    "setup": "source",
    "system": "source",
    "t": "time_s",
    "time": "time_s",
    "timestamp": "time_s",
    "timestamp_s": "time_s",
    "x_mm": "x",
    "x_norm": "x",
    "y_mm": "y",
    "y_norm": "y",
    "z_mm": "z",
    "z_norm": "z",
}

REQUIRED_COLUMNS = {"frame", "joint", "x", "y"}
SORT_COLUMNS = ["source", "domain", "trial", "frame", "joint"]


def load_pose_csv(path: str | Path) -> pd.DataFrame:
    """Load a pose CSV and return a normalized long-form DataFrame.

    The canonical format is one landmark per row per frame. The loader is
    intentionally permissive because pose exports tend to have inconsistent
    naming conventions while the analysis math needs a strict schema.
    """
    source_path = Path(path)
    df = pd.read_csv(source_path)
    if df.empty:
        raise ValueError(f"{source_path} is empty.")

    df = normalize_columns(df)
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"{source_path} is missing required columns: {missing_text}")

    if "source" not in df.columns:
        df["source"] = "source_1"
        warnings.warn("No source column found; using source_1.", stacklevel=2)
    if "trial" not in df.columns:
        df["trial"] = "trial_1"
        warnings.warn("No trial column found; using trial_1.", stacklevel=2)
    if "domain" not in df.columns:
        df["domain"] = df["source"]

    for column in ["source", "domain", "trial", "joint"]:
        df[column] = df[column].astype(str).map(normalize_label)

    df["frame"] = pd.to_numeric(df["frame"], errors="raise")
    if df["frame"].dropna().map(lambda value: float(value).is_integer()).all():
        df["frame"] = df["frame"].astype(int)

    for column in coordinate_columns(df):
        df[column] = pd.to_numeric(df[column], errors="coerce")

    if "time_s" in df.columns:
        df["time_s"] = pd.to_numeric(df["time_s"], errors="coerce")

    ordered = [column for column in SORT_COLUMNS if column in df.columns]
    return df.sort_values(ordered).reset_index(drop=True)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed: dict[str, str] = {}
    for column in df.columns:
        key = str(column).strip().lower().replace(" ", "_").replace("-", "_")
        renamed[column] = COLUMN_ALIASES.get(key, key)
    df = df.rename(columns=renamed)

    duplicate_columns = df.columns[df.columns.duplicated()].unique().tolist()
    if duplicate_columns:
        duplicates = ", ".join(duplicate_columns)
        raise ValueError(f"Column aliases collide after normalization: {duplicates}")
    return df


def normalize_label(value: object) -> str:
    return str(value).strip().lower().replace(" ", "_").replace("-", "_")


def coordinate_columns(df: pd.DataFrame) -> list[str]:
    return [column for column in ["x", "y", "z"] if column in df.columns]


def available_coordinate_columns(rows: pd.DataFrame | pd.Series) -> list[str]:
    """Return coordinate columns that contain usable values for a row group."""
    columns = coordinate_columns(rows.to_frame().T if isinstance(rows, pd.Series) else rows)
    if isinstance(rows, pd.Series):
        return [column for column in columns if pd.notna(rows[column])]
    return [column for column in columns if rows[column].notna().any()]


def write_csvs(tables: dict[str, pd.DataFrame], output_dir: str | Path) -> dict[str, Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for name, table in tables.items():
        path = out / f"{name}.csv"
        table.to_csv(path, index=False)
        written[name] = path
    return written


def require_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = set(columns).difference(df.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"Missing required columns: {missing_text}")

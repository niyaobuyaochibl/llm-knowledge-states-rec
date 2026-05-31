#!/usr/bin/env python3
"""Aggregate prepared formal results into manuscript-facing tables.

This script intentionally keeps the post-pilot formal area clean:

- Table 1 and Table 2 reuse dataset-level statistics/drift outputs from the
  validated preparation runs.
- Tables 3-5 are built only from completed formal seed directories.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd

ROOT = Path("/root/temporal_popularity_pilot")
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from temporal_popularity.reporting import ensure_dirs, markdown_table  # noqa: E402

RESULTS = ROOT / "results"
FORMAL = RESULTS / "formal"
AGG = FORMAL / "aggregate"

DATASET_LABELS = {
    "ml1m": "MovieLens-1M",
    "yelp_original_reviews": "Yelp original reviews",
}


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def read_json(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def formal_seed_dirs() -> Iterable[Path]:
    for dataset_dir in [FORMAL / "ml1m", FORMAL / "yelp"]:
        if not dataset_dir.exists():
            continue
        for seed_dir in sorted(dataset_dir.glob("seed*")):
            if seed_dir.is_dir():
                yield seed_dir


def dataset_label(config: Dict[str, object], seed_dir: Path) -> str:
    dataset = str(config.get("dataset") or seed_dir.parent.name)
    return DATASET_LABELS.get(dataset, dataset)


def seed_value(config: Dict[str, object], seed_dir: Path) -> int:
    if "seed" in config:
        return int(config["seed"])
    return int(seed_dir.name.replace("seed", ""))


def add_seed_metadata(frame: pd.DataFrame, seed_dir: Path) -> pd.DataFrame:
    if frame.empty:
        return frame
    config = read_json(seed_dir / "config.json")
    enriched = frame.copy()
    enriched.insert(0, "Seed", seed_value(config, seed_dir))
    enriched.insert(0, "Dataset", dataset_label(config, seed_dir))
    return enriched


def collect_seed_table(filename: str, preferred_filename: str | Sequence[str] | None = None) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for seed_dir in formal_seed_dirs():
        path = seed_dir / filename
        preferred = []
        if isinstance(preferred_filename, str):
            preferred = [preferred_filename]
        elif preferred_filename:
            preferred = list(preferred_filename)
        for candidate in preferred:
            if (seed_dir / candidate).exists():
                path = seed_dir / candidate
                break
        frame = read_csv(path)
        if not frame.empty:
            frames.append(add_seed_metadata(frame, seed_dir))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def summarize_numeric(frame: pd.DataFrame, keys: Sequence[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    numeric_cols = [
        col
        for col in frame.select_dtypes(include=[np.number]).columns
        if col not in set(keys) and col != "Seed"
    ]
    if not numeric_cols:
        return frame.drop_duplicates(list(keys)).reset_index(drop=True)
    grouped = frame.groupby(list(keys), dropna=False)
    counts = grouped["Seed"].nunique().rename("SeedCount").reset_index()
    mean = grouped[numeric_cols].mean().add_suffix("_mean").reset_index()
    std = grouped[numeric_cols].std(ddof=1).add_suffix("_std").reset_index()
    return counts.merge(mean, on=list(keys), how="left").merge(std, on=list(keys), how="left")


def build_table1() -> pd.DataFrame:
    ml1m = read_csv(RESULTS / "ml1m_e200" / "table1_dataset_timestamp_availability.csv")
    if not ml1m.empty and "Dataset" in ml1m.columns:
        ml1m = ml1m[ml1m["Dataset"] == "MovieLens-1M"].copy()
    yelp = read_csv(RESULTS / "yelp_day1" / "table1_dataset_timestamp_availability.csv")
    frames = [frame for frame in [ml1m, yelp] if not frame.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def build_table2() -> pd.DataFrame:
    sources = [
        (
            RESULTS / "ml1m_e200" / "table2_temporal_drift_stats.csv",
            "MovieLens-1M",
            "exact_per_user_timestamp",
        ),
        (
            RESULTS / "yelp_day1" / "table2_temporal_drift_stats.csv",
            "Yelp original reviews",
            "weighted_test_time_snapshots_200",
        ),
    ]
    frames: List[pd.DataFrame] = []
    for path, dataset, protocol in sources:
        frame = read_csv(path)
        if frame.empty:
            continue
        frame = frame.copy()
        frame.insert(0, "TemporalProtocol", protocol)
        frame.insert(0, "Dataset", dataset)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False)


def collect_run_status() -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for seed_dir in formal_seed_dirs():
        config = read_json(seed_dir / "config.json")
        manifest = read_json(seed_dir / "run_manifest.json")
        rows.append(
            {
                "Dataset": dataset_label(config, seed_dir),
                "Seed": seed_value(config, seed_dir),
                "Status": manifest.get("status", "missing_manifest"),
                "HasSplitStats": (seed_dir / "split_stats.csv").exists(),
                "HasTable3Input": (seed_dir / "test_method_metrics.csv").exists(),
                "HasTable4Input": (seed_dir / "tod_rfr.csv").exists(),
                "HasTable5Input": (seed_dir / "group_sensitivity.csv").exists(),
                "HasExtendedTables": (seed_dir / "test_method_metrics_extended.csv").exists()
                and (seed_dir / "tod_rfr_extended.csv").exists()
                and (seed_dir / "group_sensitivity_extended.csv").exists(),
                "HasFullTables": (seed_dir / "test_method_metrics_full.csv").exists()
                and (seed_dir / "tod_rfr_full.csv").exists()
                and (seed_dir / "group_sensitivity_full.csv").exists(),
                "PopCalStatus": manifest.get("popcal_status", ""),
                "XQuadStatus": manifest.get("xquad_status", ""),
                "CompletedOK": (seed_dir / "completed.ok").exists(),
            }
        )
    return pd.DataFrame(rows)


def write_report(status: pd.DataFrame, table1: pd.DataFrame, table2: pd.DataFrame) -> None:
    lines = [
        "# Formal Aggregate Report",
        "",
        f"Generated UTC: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Run Status",
        "",
        markdown_table(status) if not status.empty else "No formal seed directories found.",
        "",
        "## Dataset Stats Sources",
        "",
        markdown_table(table1) if not table1.empty else "Table 1 source data missing.",
        "",
        "## Temporal Drift Sources",
        "",
        markdown_table(table2) if not table2.empty else "Table 2 source data missing.",
        "",
        "Tables 3-5 are generated only from formal seed outputs. Empty CSVs mean no completed formal seed has produced that table input yet.",
    ]
    (AGG / "formal_aggregate_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ensure_dirs(AGG)

    table1 = build_table1()
    table2 = build_table2()
    table3_by_seed = collect_seed_table(
        "test_method_metrics.csv", ["test_method_metrics_full.csv", "test_method_metrics_extended.csv"]
    )
    table4_by_seed = collect_seed_table("tod_rfr.csv", ["tod_rfr_full.csv", "tod_rfr_extended.csv"])
    table5_by_seed = collect_seed_table(
        "group_sensitivity.csv", ["group_sensitivity_full.csv", "group_sensitivity_extended.csv"]
    )

    table3 = summarize_numeric(table3_by_seed, ["Dataset", "Method"])
    table4 = summarize_numeric(table4_by_seed, ["Dataset", "Method", "Metric", "TemporalDefinition"])
    table5 = summarize_numeric(table5_by_seed, ["Dataset", "Method", "Group"])
    status = collect_run_status()

    write_csv(table1, AGG / "table1_dataset_stats.csv")
    write_csv(table2, AGG / "table2_temporal_drift.csv")
    write_csv(table3_by_seed, AGG / "table3_static_vs_temporal_eval_by_seed.csv")
    write_csv(table3, AGG / "table3_static_vs_temporal_eval.csv")
    write_csv(table4_by_seed, AGG / "table4_tod_rfr_by_seed.csv")
    write_csv(table4, AGG / "table4_tod_rfr.csv")
    write_csv(table5_by_seed, AGG / "table5_group_sensitivity_by_seed.csv")
    write_csv(table5, AGG / "table5_group_sensitivity.csv")
    write_csv(status, AGG / "formal_run_status.csv")
    write_report(status, table1, table2)

    print(f"Wrote aggregate files under {AGG}", flush=True)


if __name__ == "__main__":
    main()

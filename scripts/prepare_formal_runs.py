#!/usr/bin/env python3
"""Create per-seed formal run directories and manifests.

This is preparation only: it writes configs/manifests but does not train models
or evaluate recommendations.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

ROOT = Path("/root/temporal_popularity_pilot")
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from temporal_popularity.data import infer_shape, read_interaction_split  # noqa: E402
from temporal_popularity.reporting import ensure_dirs, markdown_table  # noqa: E402


DATA_DIRS = {
    "ml1m": ROOT / "data/ml1m_e200",
    "yelp_original_reviews": ROOT / "data/yelp_day1",
}

FORMAL_DATASET_DIR = {
    "ml1m": "ml1m",
    "yelp_original_reviews": "yelp",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--configs",
        nargs="+",
        type=Path,
        default=[
            ROOT / "configs/formal/ml1m_formal_template.json",
            ROOT / "configs/formal/yelp_formal_template.json",
        ],
    )
    parser.add_argument("--formal-root", type=Path, default=ROOT / "results/formal")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def run_dir_for(formal_root: Path, dataset: str, seed: int) -> Path:
    return formal_root / FORMAL_DATASET_DIR[dataset] / f"seed{seed}"


def seed_config(base_config: Dict[str, object], seed: int) -> Dict[str, object]:
    config = dict(base_config)
    config["seed"] = seed
    config["seeds"] = [seed]
    return config


def prepare_one_config(config_path: Path, formal_root: Path, overwrite: bool) -> List[Dict[str, object]]:
    base_config = json.loads(config_path.read_text(encoding="utf-8"))
    dataset = str(base_config["dataset"])
    datadir = DATA_DIRS[dataset]
    train, val, test, all_events = read_interaction_split(datadir)
    n_users, n_items = infer_shape(train, val, test)
    rows: List[Dict[str, object]] = []
    for seed in base_config["seeds"]:
        outdir = run_dir_for(formal_root, dataset, int(seed))
        ensure_dirs(outdir)
        config_out = outdir / "config.json"
        manifest_out = outdir / "run_manifest.json"
        if config_out.exists() and not overwrite:
            status = "exists"
        else:
            config = seed_config(base_config, int(seed))
            config["prepared_data"] = {
                "datadir": str(datadir),
                "users": n_users,
                "items": n_items,
                "train": len(train),
                "val": len(val),
                "test": len(test),
                "all_events": len(all_events),
            }
            config_out.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
            manifest = {
                "status": "prepared_not_run",
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "dataset": dataset,
                "seed": int(seed),
                "config": str(config_out),
                "expected_outputs": [
                    "split_stats.csv",
                    "validation_all_lambda_metrics.csv",
                    "test_method_metrics.csv",
                    "test_user_level_metrics.csv",
                    "tod_rfr.csv",
                    "group_sensitivity.csv",
                ],
            }
            manifest_out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            status = "prepared"
        rows.append(
            {
                "Dataset": dataset,
                "Seed": int(seed),
                "RunDir": str(outdir),
                "Status": status,
                "Config": str(config_out),
                "Manifest": str(manifest_out),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    ensure_dirs(args.formal_root / "aggregate")
    rows: List[Dict[str, object]] = []
    for config_path in args.configs:
        rows.extend(prepare_one_config(config_path, args.formal_root, args.overwrite))
    import pandas as pd

    summary = pd.DataFrame(rows)
    summary.to_csv(args.formal_root / "aggregate/formal_run_preparation.csv", index=False)
    report = [
        "# Formal Run Preparation",
        "",
        "This preparation step wrote per-seed configs and manifests only. No models were trained.",
        "",
        markdown_table(summary),
    ]
    (args.formal_root / "aggregate/formal_run_preparation.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Done. Report: {args.formal_root / 'aggregate/formal_run_preparation.md'}", flush=True)


if __name__ == "__main__":
    main()

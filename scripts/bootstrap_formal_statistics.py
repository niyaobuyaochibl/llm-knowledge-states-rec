#!/usr/bin/env python3
"""User-level bootstrap CIs and paired tests for formal outputs."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path("/root/temporal_popularity_pilot")
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from temporal_popularity.reporting import ensure_dirs, markdown_table  # noqa: E402

FORMAL = ROOT / "results" / "formal"
OUT = FORMAL / "statistics"

DATASET_LABELS = {
    "ml1m": "MovieLens-1M",
    "yelp_original_reviews": "Yelp original reviews",
}

METHOD_METRICS = [
    "NDCG@20",
    "Recall@20",
    "Static_ARP@20",
    "Recent_ARP@20",
    "Decay_ARP@20",
    "Static_LTR@20",
    "Recent_LTR@20",
    "Decay_LTR@20",
    "Static_PCE@20",
    "Recent_PCE@20",
    "Decay_PCE@20",
]

TOD_METRICS = {
    "ARP": ("Static_ARP@20", "Decay_ARP@20", -1.0),
    "LTR": ("Static_LTR@20", "Decay_LTR@20", 1.0),
    "PCE": ("Static_PCE@20", "Decay_PCE@20", -1.0),
}

STATIC_DECAY_TESTS = {
    "ARP": ("Static_ARP@20", "Decay_ARP@20", -1.0),
    "LTR": ("Static_LTR@20", "Decay_LTR@20", 1.0),
    "PCE": ("Static_PCE@20", "Decay_PCE@20", -1.0),
}

METHOD_COMPARISON_TESTS = {
    "NDCG": ("NDCG@20", 1.0),
    "Static_ARP": ("Static_ARP@20", -1.0),
    "Decay_ARP": ("Decay_ARP@20", -1.0),
    "Static_LTR": ("Static_LTR@20", 1.0),
    "Decay_LTR": ("Decay_LTR@20", 1.0),
    "Static_PCE": ("Static_PCE@20", -1.0),
    "Decay_PCE": ("Decay_PCE@20", -1.0),
}


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


def dataset_label(config: Mapping[str, object], seed_dir: Path) -> str:
    dataset = str(config.get("dataset") or seed_dir.parent.name)
    return DATASET_LABELS.get(dataset, dataset)


def seed_value(config: Mapping[str, object], seed_dir: Path) -> int:
    if "seed" in config:
        return int(config["seed"])
    return int(seed_dir.name.replace("seed", ""))


def preferred_user_metrics_path(seed_dir: Path) -> Path:
    for name in [
        "test_user_level_metrics_full.csv",
        "test_user_level_metrics_extended.csv",
        "test_user_level_metrics.csv",
    ]:
        path = seed_dir / name
        if path.exists():
            return path
    return seed_dir / "test_user_level_metrics_full.csv"


def method_order(frame: pd.DataFrame) -> List[str]:
    return list(dict.fromkeys(frame["Method"].astype(str).tolist()))


def aligned_method_arrays(
    frame: pd.DataFrame,
    methods: Sequence[str],
    columns: Sequence[str],
) -> Tuple[np.ndarray, Dict[str, Dict[str, np.ndarray]]]:
    arrays: Dict[str, Dict[str, np.ndarray]] = {}
    reference_uids: np.ndarray | None = None
    for method in methods:
        sub = frame[frame["Method"] == method].sort_values("uid", kind="mergesort")
        uids = sub["uid"].to_numpy(np.int64)
        if reference_uids is None:
            reference_uids = uids
        elif len(uids) != len(reference_uids) or not np.array_equal(uids, reference_uids):
            raise ValueError(f"User alignment mismatch for method {method}")
        arrays[method] = {
            col: sub[col].to_numpy(np.float32, copy=True)
            for col in columns
            if col in sub.columns
        }
    if reference_uids is None:
        raise ValueError("No method rows found")
    return reference_uids, arrays


def bootstrap_matrix(
    values: np.ndarray,
    samples: int,
    seed: int,
    chunk_size: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_rows, n_cols = values.shape
    boot = np.empty((samples, n_cols), dtype=np.float32)
    for start in range(0, samples, chunk_size):
        end = min(start + chunk_size, samples)
        idx = rng.integers(0, n_rows, size=(end - start, n_rows), dtype=np.int32)
        boot[start:end] = values[idx].mean(axis=1, dtype=np.float64)
    return boot


def ci_record(values: np.ndarray) -> Dict[str, float]:
    return {
        "BootstrapMean": float(np.mean(values)),
        "BootstrapStd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "CI95Low": float(np.percentile(values, 2.5)),
        "CI95High": float(np.percentile(values, 97.5)),
    }


def rank_biserial_from_diff(diff: np.ndarray) -> float:
    clean = diff[np.isfinite(diff)]
    clean = clean[clean != 0]
    if len(clean) == 0:
        return 0.0
    ranks = pd.Series(np.abs(clean)).rank(method="average").to_numpy(np.float64)
    pos = float(ranks[clean > 0].sum())
    neg = float(ranks[clean < 0].sum())
    denom = pos + neg
    return 0.0 if denom == 0 else (pos - neg) / denom


def paired_wilcoxon(diff: np.ndarray) -> Tuple[float, float, int, float]:
    clean = diff[np.isfinite(diff)]
    clean = clean[clean != 0]
    effect = rank_biserial_from_diff(diff)
    if len(clean) == 0:
        return 0.0, 1.0, 0, effect
    ranks = pd.Series(np.abs(clean)).rank(method="average").to_numpy(np.float64)
    w_pos = float(ranks[clean > 0].sum())
    w_neg = float(ranks[clean < 0].sum())
    statistic = min(w_pos, w_neg)
    n = len(clean)
    mean = n * (n + 1) / 4.0
    _, counts = np.unique(np.abs(clean), return_counts=True)
    tie_adjustment = float(np.sum(counts**3 - counts)) / 48.0
    variance = n * (n + 1) * (2 * n + 1) / 24.0 - tie_adjustment
    if variance <= 0:
        pvalue = 1.0 if statistic == mean else 0.0
    else:
        z = (statistic - mean) / math.sqrt(variance)
        pvalue = math.erfc(abs(z) / math.sqrt(2.0))
    return float(statistic), float(pvalue), int(n), effect


def build_seed_statistics(
    dataset: str,
    seed: int,
    frame: pd.DataFrame,
    samples: int,
    chunk_size: int,
) -> Tuple[
    List[Dict[str, object]],
    List[Dict[str, object]],
    Dict[Tuple[str, str, str], np.ndarray],
    Dict[Tuple[str, str, str], np.ndarray],
    List[Dict[str, object]],
    List[Dict[str, object]],
]:
    methods = method_order(frame)
    needed_cols = sorted(set(METHOD_METRICS) | {col for pair in TOD_METRICS.values() for col in pair[:2]})
    _, arrays = aligned_method_arrays(frame, methods, needed_cols)

    stat_names: List[Tuple[str, str, str]] = []
    stat_values: List[np.ndarray] = []
    for method in methods:
        for metric in METHOD_METRICS:
            stat_names.append(("method_metric", method, metric))
            stat_values.append(arrays[method][metric])

    base = arrays["Base"]
    for method in methods:
        if method == "Base":
            continue
        for metric, (static_col, temporal_col, direction) in TOD_METRICS.items():
            static_gain = direction * arrays[method][static_col] - direction * base[static_col]
            temporal_gain = direction * arrays[method][temporal_col] - direction * base[temporal_col]
            stat_names.append(("tod_decay", method, metric))
            stat_values.append(static_gain - temporal_gain)

    values = np.column_stack(stat_values).astype(np.float32, copy=False)
    boot_seed = seed * 1009 + len(dataset) * 9173
    boot = bootstrap_matrix(values, samples=samples, seed=boot_seed, chunk_size=chunk_size)
    observed = values.mean(axis=0, dtype=np.float64)

    metric_rows: List[Dict[str, object]] = []
    tod_rows: List[Dict[str, object]] = []
    metric_boots: Dict[Tuple[str, str, str], np.ndarray] = {}
    tod_boots: Dict[Tuple[str, str, str], np.ndarray] = {}
    for idx, (kind, method, metric) in enumerate(stat_names):
        row = {
            "Dataset": dataset,
            "Seed": seed,
            "Method": method,
            "Metric": metric,
            "ObservedMean": float(observed[idx]),
            "BootstrapSamples": samples,
            **ci_record(boot[:, idx]),
        }
        if kind == "method_metric":
            metric_rows.append(row)
            metric_boots[(dataset, method, metric)] = boot[:, idx].copy()
        else:
            row["TemporalDefinition"] = "Decay"
            tod_rows.append(row)
            tod_boots[(dataset, method, metric)] = boot[:, idx].copy()

    static_decay_rows: List[Dict[str, object]] = []
    for method in methods:
        for metric, (static_col, decay_col, direction) in STATIC_DECAY_TESTS.items():
            static_q = direction * arrays[method][static_col]
            decay_q = direction * arrays[method][decay_col]
            diff = decay_q - static_q
            statistic, pvalue, n_nonzero, effect = paired_wilcoxon(diff)
            static_decay_rows.append(
                {
                    "Dataset": dataset,
                    "Seed": seed,
                    "Method": method,
                    "Metric": metric,
                    "TemporalDefinition": "Decay",
                    "Users": int(len(diff)),
                    "NonzeroPairs": n_nonzero,
                    "StaticQualityMean": float(np.mean(static_q)),
                    "TemporalQualityMean": float(np.mean(decay_q)),
                    "TemporalMinusStaticQualityMean": float(np.mean(diff)),
                    "WilcoxonStatistic": statistic,
                    "PValue": pvalue,
                    "RankBiserial": effect,
                }
            )

    method_test_rows: List[Dict[str, object]] = []
    for method in methods:
        if method == "Base":
            continue
        for metric, (col, direction) in METHOD_COMPARISON_TESTS.items():
            method_q = direction * arrays[method][col]
            base_q = direction * base[col]
            diff = method_q - base_q
            statistic, pvalue, n_nonzero, effect = paired_wilcoxon(diff)
            method_test_rows.append(
                {
                    "Dataset": dataset,
                    "Seed": seed,
                    "Method": method,
                    "ComparedTo": "Base",
                    "Metric": metric,
                    "Users": int(len(diff)),
                    "NonzeroPairs": n_nonzero,
                    "MethodQualityMean": float(np.mean(method_q)),
                    "BaseQualityMean": float(np.mean(base_q)),
                    "MethodMinusBaseQualityMean": float(np.mean(diff)),
                    "WilcoxonStatistic": statistic,
                    "PValue": pvalue,
                    "RankBiserial": effect,
                }
            )

    return metric_rows, tod_rows, metric_boots, tod_boots, static_decay_rows, method_test_rows


def summarize_bootstrap(
    rows: Sequence[Dict[str, object]],
    boots: Mapping[Tuple[str, str, str], Sequence[np.ndarray]],
    kind: str,
) -> pd.DataFrame:
    observed = defaultdict(list)
    for row in rows:
        key = (str(row["Dataset"]), str(row["Method"]), str(row["Metric"]))
        observed[key].append(float(row["ObservedMean"]))

    out_rows: List[Dict[str, object]] = []
    for key, vectors in sorted(boots.items()):
        dataset, method, metric = key
        matrix = np.vstack(vectors)
        seed_mean_boot = matrix.mean(axis=0)
        row = {
            "Dataset": dataset,
            "Method": method,
            "Metric": metric,
            "SeedCount": len(vectors),
            "ObservedMean": float(np.mean(observed[key])),
            "ObservedSeedStd": float(np.std(observed[key], ddof=1)) if len(observed[key]) > 1 else 0.0,
            **ci_record(seed_mean_boot),
        }
        if kind == "tod_decay":
            row["TemporalDefinition"] = "Decay"
        out_rows.append(row)
    return pd.DataFrame(out_rows)


def summarize_tests(frame: pd.DataFrame, keys: Sequence[str]) -> pd.DataFrame:
    if frame.empty:
        return frame
    grouped = frame.groupby(list(keys), dropna=False)
    rows: List[Dict[str, object]] = []
    for key, sub in grouped:
        key_tuple = key if isinstance(key, tuple) else (key,)
        row = dict(zip(keys, key_tuple))
        row.update(
            {
                "SeedCount": int(sub["Seed"].nunique()),
                "Users_mean": float(sub["Users"].mean()),
                "NonzeroPairs_mean": float(sub["NonzeroPairs"].mean()),
                "MeanDifference_mean": float(
                    sub[
                        "TemporalMinusStaticQualityMean"
                        if "TemporalMinusStaticQualityMean" in sub.columns
                        else "MethodMinusBaseQualityMean"
                    ].mean()
                ),
                "MeanDifference_std": float(
                    sub[
                        "TemporalMinusStaticQualityMean"
                        if "TemporalMinusStaticQualityMean" in sub.columns
                        else "MethodMinusBaseQualityMean"
                    ].std(ddof=1)
                ),
                "RankBiserial_mean": float(sub["RankBiserial"].mean()),
                "RankBiserial_std": float(sub["RankBiserial"].std(ddof=1)),
                "PValue_median": float(sub["PValue"].median()),
                "SignificantSeeds_p005": int((sub["PValue"] < 0.05).sum()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def write_report(
    method_ci: pd.DataFrame,
    tod_ci: pd.DataFrame,
    static_decay_tests: pd.DataFrame,
    method_tests: pd.DataFrame,
    samples: int,
) -> None:
    status = pd.DataFrame(
        [
            {
                "BootstrapSamples": samples,
                "MethodMetricRows": len(method_ci),
                "TODRows": len(tod_ci),
                "StaticDecayTestRows": len(static_decay_tests),
                "MethodVsBaseTestRows": len(method_tests),
                "GeneratedUTC": datetime.now(timezone.utc).isoformat(),
            }
        ]
    )

    key_tod = tod_ci[
        (tod_ci["Dataset"] == "Yelp original reviews")
        & (tod_ci["Metric"].isin(["ARP", "LTR", "PCE"]))
    ].copy()
    if not key_tod.empty:
        key_tod = key_tod.sort_values(["Metric", "Method"])[
            ["Dataset", "Method", "Metric", "ObservedMean", "CI95Low", "CI95High"]
        ]

    lines = [
        "# Formal Statistical Testing Report",
        "",
        markdown_table(status),
        "",
        "## Bootstrap TOD CI Preview",
        "",
        markdown_table(key_tod.head(20)) if not key_tod.empty else "No TOD CI rows found.",
        "",
        "## Notes",
        "",
        "- Bootstrap CIs are user-level resampling CIs, averaged across seeds for aggregate rows.",
        "- ARP and PCE tests use quality-oriented signs, so larger values mean better debiasing quality.",
        "- RankBiserial is reported as the paired-test effect size.",
    ]
    (OUT / "statistical_testing_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--chunk-size", type=int, default=5)
    args = parser.parse_args()

    ensure_dirs(OUT)

    metric_rows: List[Dict[str, object]] = []
    tod_rows: List[Dict[str, object]] = []
    static_decay_rows: List[Dict[str, object]] = []
    method_test_rows: List[Dict[str, object]] = []
    metric_boots: MutableMapping[Tuple[str, str, str], List[np.ndarray]] = defaultdict(list)
    tod_boots: MutableMapping[Tuple[str, str, str], List[np.ndarray]] = defaultdict(list)

    for seed_dir in formal_seed_dirs():
        config = read_json(seed_dir / "config.json")
        dataset = dataset_label(config, seed_dir)
        seed = seed_value(config, seed_dir)
        path = preferred_user_metrics_path(seed_dir)
        if not path.exists():
            print(f"Skipping missing user metrics: {path}", flush=True)
            continue
        print(f"Processing {dataset} seed={seed}: {path.name}", flush=True)
        frame = pd.read_csv(path)
        seed_metric, seed_tod, seed_metric_boots, seed_tod_boots, seed_static_decay, seed_method_tests = (
            build_seed_statistics(
                dataset=dataset,
                seed=seed,
                frame=frame,
                samples=args.samples,
                chunk_size=args.chunk_size,
            )
        )
        metric_rows.extend(seed_metric)
        tod_rows.extend(seed_tod)
        static_decay_rows.extend(seed_static_decay)
        method_test_rows.extend(seed_method_tests)
        for key, vector in seed_metric_boots.items():
            metric_boots[key].append(vector)
        for key, vector in seed_tod_boots.items():
            tod_boots[key].append(vector)
        print(f"Finished {dataset} seed={seed}", flush=True)

    metric_by_seed = pd.DataFrame(metric_rows)
    tod_by_seed = pd.DataFrame(tod_rows)
    static_decay_by_seed = pd.DataFrame(static_decay_rows)
    method_tests_by_seed = pd.DataFrame(method_test_rows)

    metric_ci = summarize_bootstrap(metric_rows, metric_boots, kind="method_metric")
    tod_ci = summarize_bootstrap(tod_rows, tod_boots, kind="tod_decay")
    static_decay_summary = summarize_tests(
        static_decay_by_seed,
        ["Dataset", "Method", "Metric", "TemporalDefinition"],
    )
    method_tests_summary = summarize_tests(
        method_tests_by_seed,
        ["Dataset", "Method", "ComparedTo", "Metric"],
    )

    metric_by_seed.to_csv(OUT / "bootstrap_method_metric_ci_by_seed.csv", index=False)
    metric_ci.to_csv(OUT / "bootstrap_method_metric_ci.csv", index=False)
    tod_by_seed.to_csv(OUT / "bootstrap_tod_ci_by_seed.csv", index=False)
    tod_ci.to_csv(OUT / "bootstrap_tod_ci.csv", index=False)
    static_decay_by_seed.to_csv(OUT / "wilcoxon_static_vs_decay_by_seed.csv", index=False)
    static_decay_summary.to_csv(OUT / "wilcoxon_static_vs_decay.csv", index=False)
    method_tests_by_seed.to_csv(OUT / "wilcoxon_method_vs_base_by_seed.csv", index=False)
    method_tests_summary.to_csv(OUT / "wilcoxon_method_vs_base.csv", index=False)
    write_report(metric_ci, tod_ci, static_decay_summary, method_tests_summary, args.samples)
    print(f"Wrote statistical testing files under {OUT}", flush=True)


if __name__ == "__main__":
    main()

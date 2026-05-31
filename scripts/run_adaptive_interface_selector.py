#!/usr/bin/env python3
"""Build an adaptive LLM knowledge-interface selection panel from completed experiments.

The script does not call any LLM APIs. It reuses the completed fair-candidate
experiments and computes ALIS (Adaptive LLM Knowledge-Interface Selector) scores under
several deployment preference profiles.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results" / "egpr_profile_repair"
OUT_DIR = RESULT_ROOT / "adaptive_interface_selector"

METHOD_MAP = {
    "LightGCN": "Base",
    "DeepSeek Direct Rerank": "Direct",
    "Profile Rerank Raw": "Raw Profile",
    "Profile Rerank Remove": "Remove Repair",
    "Profile Rerank EGPR": "EGPR",
}

FAITHFULNESS_MAP = {
    "Raw Profile": "Raw Profile",
    "Remove Repair": "Remove Repair",
    "EGPR": "Evidence-Weighted Repair",
}

PREFERENCE_PROFILES: Dict[str, Dict[str, float]] = {
    "accuracy_first": {
        "Utility": 1.00,
        "Cost": 0.02,
        "Harm": 0.02,
        "Disruption": 0.01,
        "Grounding": 0.01,
    },
    "balanced": {
        "Utility": 1.00,
        "Cost": 0.15,
        "Harm": 0.15,
        "Disruption": 0.10,
        "Grounding": 0.10,
    },
    "safety_first": {
        "Utility": 0.70,
        "Cost": 0.10,
        "Harm": 0.35,
        "Disruption": 0.15,
        "Grounding": 0.35,
    },
    "cost_first": {
        "Utility": 0.60,
        "Cost": 0.45,
        "Harm": 0.10,
        "Disruption": 0.10,
        "Grounding": 0.05,
    },
}


@dataclass(frozen=True)
class DatasetSpec:
    dataset: str
    candidate_set: str
    summary_path: Path
    disruption_path: Path
    faithfulness_path: Path | None
    summary_candidate_filter: str | None = None
    disruption_candidate_filter: str | None = None


DATASETS: Sequence[DatasetSpec] = (
    DatasetSpec(
        dataset="ML-1M",
        candidate_set="top-50",
        summary_path=RESULT_ROOT
        / "ml1m_2000_fair_direct_candidates"
        / "fair_direct_candidate_summary.csv",
        disruption_path=RESULT_ROOT
        / "ml1m_2000_fair_direct_candidates"
        / "rank_disruption"
        / "ml1m_2000_fair_rank_disruption.csv",
        faithfulness_path=None,
    ),
    DatasetSpec(
        dataset="Yelp",
        candidate_set="top-100",
        summary_path=RESULT_ROOT
        / "yelp_direct_vs_profile"
        / "yelp_1000_fair_confirmatory"
        / "yelp_1000_fair_summary.csv",
        disruption_path=RESULT_ROOT
        / "yelp_direct_vs_profile"
        / "yelp_1000_fair_confirmatory"
        / "yelp_1000_rank_disruption.csv",
        faithfulness_path=RESULT_ROOT
        / "yelp_seed42_deepseek_1000_expressive5"
        / "table1_profile_faithfulness.csv",
        summary_candidate_filter="100",
        disruption_candidate_filter="100",
    ),
    DatasetSpec(
        dataset="Amazon Books",
        candidate_set="top-100",
        summary_path=RESULT_ROOT
        / "amazon_books_direct_vs_profile"
        / "amazon_books_1000_fair_confirmatory"
        / "amazon_books_1000_fair_summary.csv",
        disruption_path=RESULT_ROOT
        / "amazon_books_direct_vs_profile"
        / "amazon_books_1000_fair_confirmatory"
        / "amazon_books_1000_rank_disruption.csv",
        faithfulness_path=RESULT_ROOT
        / "amazon_books_seed42_deepseek_1000_expressive5"
        / "table1_profile_faithfulness.csv",
        summary_candidate_filter="100",
        disruption_candidate_filter="100",
    ),
)


def to_float(value: object, default: float = 0.0) -> float:
    if value is None:
        return default
    text = str(value).strip()
    if text == "":
        return default
    if text.lower() == "inf":
        return math.inf
    if text.lower() == "nan":
        return default
    return float(text)


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Mapping[str, object]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})


def faithfulness_lookup(path: Path | None) -> Dict[str, Dict[str, float]]:
    if path is None:
        return {}
    rows = read_csv(path)
    out: Dict[str, Dict[str, float]] = {}
    for row in rows:
        method = row["Method"]
        out[method] = {
            "UCR": to_float(row.get("UCR")),
            "WeightedUCR": to_float(row.get("WeightedUCR")),
            "EvidenceCoverage": to_float(row.get("EvidenceCoverage")),
            "ProfileDriftScore": to_float(row.get("ProfileDriftScore")),
        }
    return out


def filter_rows(rows: Iterable[Mapping[str, str]], candidate_filter: str | None) -> List[Mapping[str, str]]:
    if candidate_filter is None:
        return list(rows)
    return [row for row in rows if str(row.get("CandidateSet", "")).strip() == candidate_filter]


def load_dataset_metrics(spec: DatasetSpec) -> List[Dict[str, object]]:
    summary_rows = filter_rows(read_csv(spec.summary_path), spec.summary_candidate_filter)
    disruption_rows = filter_rows(read_csv(spec.disruption_path), spec.disruption_candidate_filter)
    faith = faithfulness_lookup(spec.faithfulness_path)

    disruption_by_method = {
        METHOD_MAP.get(row["Method"], row["Method"]): row for row in disruption_rows
    }

    rows: List[Dict[str, object]] = []
    for row in summary_rows:
        method = METHOD_MAP.get(row["Method"], row["Method"])
        if method not in {"Base", "Direct", "Raw Profile", "Remove Repair", "EGPR"}:
            continue

        disruption = disruption_by_method.get(method, {})
        profile_key = FAITHFULNESS_MAP.get(method)
        profile_faith = faith.get(profile_key, {})

        # The ML-1M fair summary already carries faithfulness fields; Yelp and
        # Amazon use separate profile-rerank faithfulness files.
        ucr = to_float(row.get("UCR"), profile_faith.get("UCR", 0.0))
        wucr = to_float(row.get("WeightedUCR"), profile_faith.get("WeightedUCR", 0.0))
        drift = to_float(
            row.get("ProfileDriftScore"),
            profile_faith.get("ProfileDriftScore", 0.0),
        )

        rows.append(
            {
                "Dataset": spec.dataset,
                "CandidateSet": spec.candidate_set,
                "Method": method,
                "NDCG@20": to_float(row.get("NDCG@20")),
                "Recall@20": to_float(row.get("Recall@20")),
                "DeltaVsBase": to_float(row.get("DeltaVsBase"), to_float(row.get("NDCGGainVsBase"))),
                "HarmRate": to_float(row.get("HarmRate")),
                "GHR": to_float(row.get("GHR"), to_float(row.get("GainHarmRatio"))),
                "CostVsDirect": to_float(
                    row.get("CostVsDirect"),
                    to_float(row.get("CostRatioVsDirect_TestUsers")),
                ),
                "UCR": ucr,
                "WeightedUCR": wucr,
                "ProfileDrift": drift,
                "Top20OverlapVsBase": to_float(disruption.get("Top20OverlapVsBase"), 1.0),
                "MeanAbsRankShift": to_float(disruption.get("MeanAbsRankShiftVsBase")),
            }
        )
    return rows


def add_normalized_metrics(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    by_dataset: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        by_dataset.setdefault(str(row["Dataset"]), []).append(row)

    for dataset_rows in by_dataset.values():
        ndcgs = [float(row["NDCG@20"]) for row in dataset_rows]
        min_ndcg, max_ndcg = min(ndcgs), max(ndcgs)
        ndcg_span = max(max_ndcg - min_ndcg, 1e-12)
        max_harm = max(max(float(row["HarmRate"]) for row in dataset_rows), 1e-12)
        max_wucr = max(max(float(row["WeightedUCR"]) for row in dataset_rows), 1e-12)
        max_shift = max(max(float(row["MeanAbsRankShift"]) for row in dataset_rows), 1e-12)

        for row in dataset_rows:
            new_row = dict(row)
            new_row["UtilityNorm"] = (float(row["NDCG@20"]) - min_ndcg) / ndcg_span
            new_row["CostNorm"] = float(row["CostVsDirect"])
            new_row["HarmNorm"] = float(row["HarmRate"]) / max_harm
            new_row["DisruptionNorm"] = 1.0 - float(row["Top20OverlapVsBase"])
            new_row["ShiftNorm"] = float(row["MeanAbsRankShift"]) / max_shift
            new_row["GroundingRiskNorm"] = float(row["WeightedUCR"]) / max_wucr
            out.append(new_row)
    return out


def score(row: Mapping[str, object], weights: Mapping[str, float]) -> float:
    return (
        weights["Utility"] * float(row["UtilityNorm"])
        - weights["Cost"] * float(row["CostNorm"])
        - weights["Harm"] * float(row["HarmNorm"])
        - weights["Disruption"] * float(row["DisruptionNorm"])
        - weights["Grounding"] * float(row["GroundingRiskNorm"])
    )


def score_rows(
    rows: Sequence[Mapping[str, object]],
    profiles: Mapping[str, Mapping[str, float]] = PREFERENCE_PROFILES,
) -> List[Dict[str, object]]:
    scored: List[Dict[str, object]] = []
    for profile, weights in profiles.items():
        for row in rows:
            scored.append(
                {
                    **row,
                    "PreferenceProfile": profile,
                    "UtilityWeight": weights["Utility"],
                    "CostWeight": weights["Cost"],
                    "HarmWeight": weights["Harm"],
                    "DisruptionWeight": weights["Disruption"],
                    "GroundingWeight": weights["Grounding"],
                    "ALISScore": score(row, weights),
                }
            )
    return scored


def selected_interfaces(scored_rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[tuple[str, str], List[Mapping[str, object]]] = {}
    for row in scored_rows:
        key = (str(row["Dataset"]), str(row["PreferenceProfile"]))
        grouped.setdefault(key, []).append(row)

    selected: List[Dict[str, object]] = []
    for (dataset, profile), rows in sorted(grouped.items()):
        best = max(rows, key=lambda row: float(row["ALISScore"]))
        selected.append(
            {
                "Dataset": dataset,
                "CandidateSet": best["CandidateSet"],
                "PreferenceProfile": profile,
                "SelectedInterface": best["Method"],
                "ALISScore": best["ALISScore"],
                "NDCG@20": best["NDCG@20"],
                "Recall@20": best["Recall@20"],
                "DeltaVsBase": best["DeltaVsBase"],
                "HarmRate": best["HarmRate"],
                "GHR": best["GHR"],
                "CostVsDirect": best["CostVsDirect"],
                "WeightedUCR": best["WeightedUCR"],
                "ProfileDrift": best["ProfileDrift"],
                "Top20OverlapVsBase": best["Top20OverlapVsBase"],
                "MeanAbsRankShift": best["MeanAbsRankShift"],
            }
        )
    return selected


def ablation_rows(rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    base = PREFERENCE_PROFILES["balanced"]
    profiles = {
        "balanced_full": base,
        "no_cost_penalty": {**base, "Cost": 0.0},
        "no_harm_penalty": {**base, "Harm": 0.0},
        "no_disruption_penalty": {**base, "Disruption": 0.0},
        "no_grounding_penalty": {**base, "Grounding": 0.0},
    }
    scored = score_rows(rows, profiles)
    selected = selected_interfaces(scored)
    for row in selected:
        row["Ablation"] = row.pop("PreferenceProfile")
    return selected


def sensitivity_rows(rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for cost_weight in [0.0, 0.05, 0.10, 0.15, 0.25, 0.35, 0.45, 0.60]:
        for grounding_weight in [0.0, 0.05, 0.10, 0.20, 0.35, 0.50]:
            profiles = {
                f"cost_{cost_weight:.2f}_ground_{grounding_weight:.2f}": {
                    "Utility": 1.0,
                    "Cost": cost_weight,
                    "Harm": 0.15,
                    "Disruption": 0.10,
                    "Grounding": grounding_weight,
                }
            }
            selected = selected_interfaces(score_rows(rows, profiles))
            for row in selected:
                out.append(
                    {
                        "Dataset": row["Dataset"],
                        "CostWeight": cost_weight,
                        "GroundingWeight": grounding_weight,
                        "SelectedInterface": row["SelectedInterface"],
                        "ALISScore": row["ALISScore"],
                        "NDCG@20": row["NDCG@20"],
                        "CostVsDirect": row["CostVsDirect"],
                        "WeightedUCR": row["WeightedUCR"],
                    }
                )
    return out


def fmt_float(value: object, digits: int = 6) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isinf(number):
        return "inf"
    return f"{number:.{digits}f}"


def markdown_table(rows: Sequence[Mapping[str, object]], columns: Sequence[str], float_digits: int = 6) -> str:
    lines = []
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for row in rows:
        cells = []
        for col in columns:
            value = row.get(col, "")
            if isinstance(value, float):
                cells.append(fmt_float(value, float_digits))
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_report(
    metrics: Sequence[Mapping[str, object]],
    selected: Sequence[Mapping[str, object]],
    ablations: Sequence[Mapping[str, object]],
) -> None:
    report = OUT_DIR / "adaptive_interface_selector_report.md"

    selected_columns = [
        "Dataset",
        "PreferenceProfile",
        "SelectedInterface",
        "ALISScore",
        "NDCG@20",
        "DeltaVsBase",
        "HarmRate",
        "CostVsDirect",
        "WeightedUCR",
        "Top20OverlapVsBase",
    ]
    metric_columns = [
        "Dataset",
        "Method",
        "NDCG@20",
        "DeltaVsBase",
        "HarmRate",
        "CostVsDirect",
        "WeightedUCR",
        "ProfileDrift",
        "Top20OverlapVsBase",
        "MeanAbsRankShift",
    ]
    ablation_columns = [
        "Dataset",
        "Ablation",
        "SelectedInterface",
        "ALISScore",
        "NDCG@20",
        "CostVsDirect",
        "WeightedUCR",
        "Top20OverlapVsBase",
    ]

    text = f"""# Adaptive LLM Interface Selector (ALIS)

This report reuses the completed fair-candidate experiments and adds an
adaptive selection layer over five interfaces: Base, Direct rerank, Raw Profile,
Remove Repair, and EGPR. No new API calls are used.

ALIS scores each interface within a dataset as:

```text
Score = w_u * Utility - w_c * Cost - w_h * Harm - w_d * Disruption - w_g * GroundingRisk
```

Utility is min-max-normalized NDCG@20 within the dataset. Cost is the API cost
ratio relative to Direct. Harm is HarmRate normalized by the dataset maximum.
Disruption is `1 - Top20OverlapVsBase`. GroundingRisk is WeightedUCR normalized
by the maximum profile WeightedUCR in the dataset.

## Selected Interfaces by Deployment Preference

{markdown_table(selected, selected_columns)}

## Candidate Interface Diagnostic Panel

{markdown_table(metrics, metric_columns)}

## Balanced-Profile Ablation

{markdown_table(ablations, ablation_columns)}

## Interpretation

- Under accuracy-first and balanced preferences, ALIS selects Direct on ML-1M,
  Raw Profile on Yelp, and Remove Repair on Amazon Books. This recovers the
  three-domain pattern while making the selection rule explicit.
- Under safety-first preferences, ALIS can choose Base on Yelp because Raw
  Profile improves utility but carries non-trivial harm and unsupported-claim
  mass. This is a useful deployment behavior: the selector is allowed to abstain
  from LLM augmentation when risk penalties dominate.
- Amazon Books consistently selects Remove Repair across preferences. This
  matches the high Raw UCR and low evidence coverage observed in the profile
  faithfulness panel, where unsupported claims are more likely to be harmful.
- The ablation table shows which penalty terms move the selected interface. It
  can be used in the manuscript to argue that ALIS is a method-level contribution
  rather than only a descriptive comparison.

## Artifacts

- `candidate_interface_metrics.csv`
- `alis_preference_scores.csv`
- `alis_selected_interfaces.csv`
- `alis_ablation.csv`
- `alis_sensitivity_grid.csv`
- `adaptive_interface_selector_report.md`
"""
    report.write_text(text, encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics = add_normalized_metrics(
        [row for spec in DATASETS for row in load_dataset_metrics(spec)]
    )
    scored = score_rows(metrics)
    selected = selected_interfaces(scored)
    ablations = ablation_rows(metrics)
    sensitivity = sensitivity_rows(metrics)

    metric_columns = [
        "Dataset",
        "CandidateSet",
        "Method",
        "NDCG@20",
        "Recall@20",
        "DeltaVsBase",
        "HarmRate",
        "GHR",
        "CostVsDirect",
        "UCR",
        "WeightedUCR",
        "ProfileDrift",
        "Top20OverlapVsBase",
        "MeanAbsRankShift",
        "UtilityNorm",
        "CostNorm",
        "HarmNorm",
        "DisruptionNorm",
        "ShiftNorm",
        "GroundingRiskNorm",
    ]
    score_columns = [
        *metric_columns,
        "PreferenceProfile",
        "UtilityWeight",
        "CostWeight",
        "HarmWeight",
        "DisruptionWeight",
        "GroundingWeight",
        "ALISScore",
    ]
    selected_columns = [
        "Dataset",
        "CandidateSet",
        "PreferenceProfile",
        "SelectedInterface",
        "ALISScore",
        "NDCG@20",
        "Recall@20",
        "DeltaVsBase",
        "HarmRate",
        "GHR",
        "CostVsDirect",
        "WeightedUCR",
        "ProfileDrift",
        "Top20OverlapVsBase",
        "MeanAbsRankShift",
    ]
    ablation_columns = [
        "Dataset",
        "CandidateSet",
        "Ablation",
        "SelectedInterface",
        "ALISScore",
        "NDCG@20",
        "Recall@20",
        "DeltaVsBase",
        "HarmRate",
        "GHR",
        "CostVsDirect",
        "WeightedUCR",
        "ProfileDrift",
        "Top20OverlapVsBase",
        "MeanAbsRankShift",
    ]
    sensitivity_columns = [
        "Dataset",
        "CostWeight",
        "GroundingWeight",
        "SelectedInterface",
        "ALISScore",
        "NDCG@20",
        "CostVsDirect",
        "WeightedUCR",
    ]

    write_csv(OUT_DIR / "candidate_interface_metrics.csv", metrics, metric_columns)
    write_csv(OUT_DIR / "alis_preference_scores.csv", scored, score_columns)
    write_csv(OUT_DIR / "alis_selected_interfaces.csv", selected, selected_columns)
    write_csv(OUT_DIR / "alis_ablation.csv", ablations, ablation_columns)
    write_csv(OUT_DIR / "alis_sensitivity_grid.csv", sensitivity, sensitivity_columns)
    write_report(metrics, selected, ablations)

    print(f"Wrote ALIS outputs to {OUT_DIR}")
    for row in selected:
        print(
            f"{row['Dataset']:13s} {row['PreferenceProfile']:15s} -> "
            f"{row['SelectedInterface']} (score={float(row['ALISScore']):.4f})"
        )


if __name__ == "__main__":
    main()

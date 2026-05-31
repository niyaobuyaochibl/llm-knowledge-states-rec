#!/usr/bin/env python3
"""Evaluate cached ML-1M DeepSeek profiles over a SASRec backbone."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

ROOT = Path("/root/temporal_popularity_pilot")
SRC = ROOT / "src"
SCRIPT_DIR = ROOT / "scripts"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SCRIPT_DIR))

from temporal_popularity.data import build_exclude_lists, build_user_histories, infer_shape, read_interaction_split  # noqa: E402
from temporal_popularity.eval import topk_indices  # noqa: E402
from temporal_popularity.sequential import (  # noqa: E402
    SASRec,
    build_ordered_user_sequences,
    sequence_rows_for_users,
)
from run_egpr_profile_repair_pilot import (  # noqa: E402
    CandidateBatch,
    ClaimRecord,
    evaluate_methods,
    profile_faithfulness,
)
from run_llm_selective_invocation_pilot import device_from_arg, ensure_dirs, read_item_metadata  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", type=Path, default=ROOT / "data/ml1m")
    parser.add_argument("--movies-path", type=Path, default=Path("/root/autodl-tmp/projects_archive/MGPrompt-CDR/data/raw/ml-1m/movies.dat"))
    parser.add_argument("--profile-run-dir", type=Path, default=ROOT / "results/egpr_profile_repair/ml1m_seed42_deepseek_2000_expressive5")
    parser.add_argument("--sasrec-ckpt", type=Path, default=ROOT / "results/formal/backbone_robustness/ml1m/seed42/sasrec/sasrec.pt")
    parser.add_argument("--outdir", type=Path, default=ROOT / "results/egpr_profile_repair/ml1m_seed42_sasrec_deepseek_2000_expressive5")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--top-candidates", type=int, default=100)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--history-limit", type=int, default=20)
    parser.add_argument("--lambda-grid", nargs="+", type=float, default=[0.05, 0.1, 0.2, 0.3, 0.5, 1.0])
    return parser.parse_args()


def load_profile_sample(profile_run_dir: Path, split: str) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(profile_run_dir / f"candidates_lightgcn_{split}.npz")
    return data["users"].astype(np.int64), data["targets"].astype(np.int64)


def load_claim_records(path: Path) -> Dict[int, List[ClaimRecord]]:
    records: Dict[int, List[ClaimRecord]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rec = ClaimRecord(
                uid=int(row["uid"]),
                claim_id=int(row["claim_id"]),
                claim=str(row["claim"]),
                claim_type=str(row["claim_type"]),
                confidence=float(row.get("confidence", 0.0) or 0.0),
                support_count=int(row["support_count"]),
                support_score=float(row["support_score"]),
                support_weight=float(row["support_weight"]),
                status=str(row["status"]),
                supporting_items=[int(x) for x in row.get("supporting_items", [])],
            )
            records.setdefault(rec.uid, []).append(rec)
    return records


def load_sasrec_model(checkpoint_path: Path, n_items: int, device: torch.device) -> SASRec:
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    model = SASRec(n_items=n_items, max_len=50, embedding_dim=64, layers=2, heads=2, dropout=0.2).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def top_candidates_for_scores(score_vector: np.ndarray, excluded: np.ndarray, topn: int) -> Tuple[np.ndarray, np.ndarray]:
    scores = score_vector.astype(np.float32).copy()
    if len(excluded):
        scores[excluded] = -np.inf
    idx = topk_indices(scores, topn)
    chosen_scores = scores[idx].astype(np.float32)
    finite = np.isfinite(chosen_scores)
    if not finite.all():
        fill = float(chosen_scores[finite].min() - 1.0) if finite.any() else 0.0
        chosen_scores = np.where(finite, chosen_scores, fill).astype(np.float32)
    return idx.astype(np.int64), chosen_scores


def generate_sasrec_batch(
    users: np.ndarray,
    targets: np.ndarray,
    exclude_lists: Sequence[np.ndarray],
    sequences: Sequence[np.ndarray],
    model: SASRec,
    n_items: int,
    topn: int,
    batch_size: int,
    device: torch.device,
    split_name: str,
) -> CandidateBatch:
    candidates = np.zeros((len(users), topn), dtype=np.int64)
    scores = np.zeros((len(users), topn), dtype=np.float32)
    for start in range(0, len(users), batch_size):
        end = min(start + batch_size, len(users))
        batch_users = users[start:end]
        seq_rows = sequence_rows_for_users(sequences, batch_users, model.max_len)
        seq_t = torch.as_tensor(seq_rows, dtype=torch.long, device=device)
        with torch.no_grad():
            score_batch = model.score_all(seq_t).detach().cpu().numpy().astype(np.float32)
        for local, uid_np in enumerate(batch_users):
            uid = int(uid_np)
            candidates[start + local], scores[start + local] = top_candidates_for_scores(
                score_batch[local, :n_items], exclude_lists[uid], topn
            )
        print(f"sasrec {split_name} candidates {end}/{len(users)}", flush=True)
    return CandidateBatch(users=users, targets=targets, candidates=candidates, scores=scores, split_name=split_name)


def save_candidates(outdir: Path, batch: CandidateBatch) -> None:
    np.savez_compressed(
        outdir / f"candidates_sasrec_{batch.split_name}.npz",
        users=batch.users,
        targets=batch.targets,
        candidates=batch.candidates,
        scores=batch.scores,
    )


def markdown_table(frame: pd.DataFrame) -> str:
    display = frame.copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(lambda value: "inf" if np.isinf(value) else f"{value:.6f}")
        else:
            display[col] = display[col].map(lambda value: "" if pd.isna(value) else str(value))
    headers = [str(col) for col in display.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in display.values.tolist():
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def write_report(outdir: Path, faith: pd.DataFrame, perf: pd.DataFrame, rel: pd.DataFrame, lambdas: pd.DataFrame, args: argparse.Namespace) -> None:
    lines = [
        "# ML-1M SASRec Profile Rerank Backbone Check",
        "",
        f"Profile source: `{args.profile_run_dir}`.",
        f"Backbone: SASRec checkpoint `{args.sasrec_ckpt}`. Candidate set: top-{args.top_candidates}. Output: top-{args.topk}.",
        "No additional LLM calls are used; cached DeepSeek profiles and claim-support records are reused.",
        "",
        "## Profile Faithfulness",
        "",
        markdown_table(faith),
        "",
        "## Recommendation Performance",
        "",
        markdown_table(perf),
        "",
        "## Reliability",
        "",
        markdown_table(rel),
        "",
        "## Lambda Validation",
        "",
        markdown_table(lambdas),
        "",
        "## Interpretation",
        "",
        "- This run isolates backbone sensitivity by replacing LightGCN candidates with SASRec candidates while keeping the same users, targets, profiles, and claim-support records.",
        "- Direct LLM reranking is intentionally not rerun here; this is a no-API profile-before-ranking backbone check.",
        "",
        "## Artifacts",
        "",
        "- `candidates_sasrec_val.npz`",
        "- `candidates_sasrec_test.npz`",
        "- `table1_profile_faithfulness.csv`",
        "- `table2_recommendation_performance.csv`",
        "- `table3_reliability.csv`",
        "- `table4_lambda_validation.csv`",
        "- `run_manifest.json`",
    ]
    (outdir / "ml1m_sasrec_profile_rerank_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    ensure_dirs(args.outdir)
    train, val, test, _ = read_interaction_split(args.datadir)
    n_users, n_items = infer_shape(train, val, test)
    train_histories = build_user_histories(train, n_users)
    val_exclude = build_exclude_lists(train_histories, None, n_users)
    test_exclude = build_exclude_lists(train_histories, val, n_users)
    ordered_train = build_ordered_user_sequences(train, n_users)
    ordered_test = build_ordered_user_sequences(pd.concat([train, val], ignore_index=True), n_users)
    meta = read_item_metadata(args.movies_path, args.datadir / "mappings.json", n_items)
    records = load_claim_records(args.profile_run_dir / "claim_support.jsonl")

    val_users, val_targets = load_profile_sample(args.profile_run_dir, "val")
    test_users, test_targets = load_profile_sample(args.profile_run_dir, "test")
    device = device_from_arg(args.device)
    model = load_sasrec_model(args.sasrec_ckpt, n_items, device)
    val_batch = generate_sasrec_batch(
        val_users, val_targets, val_exclude, ordered_train, model, n_items, args.top_candidates, args.eval_batch_size, device, "val"
    )
    test_batch = generate_sasrec_batch(
        test_users, test_targets, test_exclude, ordered_test, model, n_items, args.top_candidates, args.eval_batch_size, device, "test"
    )
    save_candidates(args.outdir, val_batch)
    save_candidates(args.outdir, test_batch)

    faith = profile_faithfulness(records, ordered_train, meta, args.history_limit)
    perf, rel, lambdas, per_user = evaluate_methods(
        val_batch, test_batch, records, meta, args.lambda_grid, args.topk
    )
    perf.loc[perf["Method"] == "LightGCN", "Method"] = "SASRec"
    perf["Method"] = perf["Method"].str.replace("LightGCN +", "SASRec +", regex=False)
    rel.loc[rel["Method"] == "LightGCN", "Method"] = "SASRec"
    rel["Method"] = rel["Method"].str.replace("LightGCN +", "SASRec +", regex=False)
    lambdas["Method"] = lambdas["Method"].str.replace("LightGCN +", "SASRec +", regex=False)

    faith.to_csv(args.outdir / "table1_profile_faithfulness.csv", index=False)
    perf.to_csv(args.outdir / "table2_recommendation_performance.csv", index=False)
    rel.to_csv(args.outdir / "table3_reliability.csv", index=False)
    lambdas.to_csv(args.outdir / "table4_lambda_validation.csv", index=False)
    for method_name, metrics in per_user.items():
        safe = re.sub(r"[^A-Za-z0-9]+", "_", method_name.replace("LightGCN", "SASRec")).strip("_").lower()
        frame = metrics.copy()
        frame.insert(0, "target", test_batch.targets.astype(int))
        frame.insert(0, "uid", test_batch.users.astype(int))
        frame.to_csv(args.outdir / f"per_user_{safe}.csv", index=False)

    manifest = {
        "status": "completed",
        "profile_run_dir": str(args.profile_run_dir),
        "sasrec_ckpt": str(args.sasrec_ckpt),
        "users_val": int(len(val_users)),
        "users_test": int(len(test_users)),
        "top_candidates": int(args.top_candidates),
        "topk": int(args.topk),
        "lambda_grid": [float(x) for x in args.lambda_grid],
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (args.outdir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    write_report(args.outdir, faith, perf, rel, lambdas, args)
    print(f"Done. Report: {args.outdir / 'ml1m_sasrec_profile_rerank_report.md'}", flush=True)


if __name__ == "__main__":
    main()

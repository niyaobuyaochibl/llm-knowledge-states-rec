#!/usr/bin/env python3
"""Direct DeepSeek reranking for Amazon Books over cached LightGCN candidates."""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path("/root/temporal_popularity_pilot")
SRC = ROOT / "src"
SCRIPT_DIR = ROOT / "scripts"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SCRIPT_DIR))

from temporal_popularity.data import read_interaction_split  # noqa: E402
from run_amazon_books_profile_rerank_pilot import (  # noqa: E402
    BookMeta,
    history_lines,
    load_metadata,
    metrics_from_ranked,
    reliability,
    summary,
)
from run_egpr_profile_repair_pilot import build_ordered_histories  # noqa: E402
from run_llm_selective_invocation_pilot import ensure_dirs, estimate_tokens  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", type=Path, default=ROOT / "data/amazon_books_subset")
    parser.add_argument("--candidate-run-dir", type=Path, default=ROOT / "results/egpr_profile_repair/amazon_books_seed42_lightgcn_1000")
    parser.add_argument("--outdir", type=Path, default=ROOT / "results/egpr_profile_repair/amazon_books_direct_vs_profile/direct_rerank_1000_top50")
    parser.add_argument("--provider", choices=["deepseek", "openai"], default="deepseek")
    parser.add_argument("--model", default=None)
    parser.add_argument("--api-url", default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--api-timeout", type=float, default=90.0)
    parser.add_argument("--api-max-retries", type=int, default=12)
    parser.add_argument("--api-max-output-tokens", type=int, default=900)
    parser.add_argument("--history-limit", type=int, default=20)
    parser.add_argument("--direct-candidates", type=int, default=50)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--input-price-per-1m", type=float, default=None)
    parser.add_argument("--output-price-per-1m", type=float, default=None)
    args = parser.parse_args()
    if args.model is None:
        args.model = "deepseek-v4-flash" if args.provider == "deepseek" else "gpt-4.1-mini"
    if args.input_price_per_1m is None:
        args.input_price_per_1m = 0.14 if args.provider == "deepseek" else 0.40
    if args.output_price_per_1m is None:
        args.output_price_per_1m = 0.28 if args.provider == "deepseek" else 1.60
    return args


def load_test_batch(candidate_run_dir: Path, direct_candidates: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(candidate_run_dir / "candidates_lightgcn_test.npz")
    return (
        data["users"].astype(np.int64),
        data["targets"].astype(np.int64),
        data["candidates"][:, :direct_candidates].astype(np.int64),
    )


def candidate_lines(candidates: Sequence[int], meta: BookMeta) -> List[str]:
    rows: List[str] = []
    for pos, iid_np in enumerate(candidates, start=1):
        iid = int(iid_np)
        cats = ", ".join(meta.categories[iid][:5]) if meta.categories[iid] else "Unknown"
        rows.append(f"{pos}. {meta.titles[iid]} | {cats}")
    return rows


def build_prompt(uid: int, candidates: Sequence[int], histories: Sequence[np.ndarray], meta: BookMeta, history_limit: int) -> str:
    lines = [
        "Given the user's Amazon Books interaction history and candidate books, rerank the candidates by predicted user preference.",
        "Return valid JSON only with schema {\"ranking\":[candidate_number, ...]}.",
        "Rules:",
        "1. Use each candidate number at most once.",
        "2. Rank all candidates from most to least relevant.",
        "3. Do not invent candidate numbers.",
        "4. Use the history and candidate metadata only.",
        "",
        f"User: U{uid}",
        "User history:",
    ]
    lines.extend(history_lines(int(uid), histories, meta, history_limit) or ["None"])
    lines.extend(["", "Candidates:"])
    lines.extend(candidate_lines(candidates, meta))
    return "\n".join(lines)


class DirectReranker:
    def __init__(self, args: argparse.Namespace) -> None:
        env_name = "DEEPSEEK_API_KEY" if args.provider == "deepseek" else "OPENAI_API_KEY"
        api_key = os.environ.get(env_name)
        if not api_key:
            raise SystemExit(f"{env_name} is not set.")
        self.provider = args.provider
        self.model = args.model
        self.api_url = args.api_url or ("https://api.deepseek.com/chat/completions" if args.provider == "deepseek" else "https://api.openai.com/v1/chat/completions")
        self.timeout = args.api_timeout
        self.max_retries = args.api_max_retries
        self.max_output_tokens = args.api_max_output_tokens
        self.api_key = api_key
        self.cache_dir = args.cache_dir or (args.outdir / "direct_api_cache")
        ensure_dirs(self.cache_dir)

    def _cache_path(self, prompt: str) -> Path:
        key = hashlib.sha256((self.provider + "\n" + self.model + "\n" + prompt).encode("utf-8")).hexdigest()
        return self.cache_dir / f"{key}.json"

    def _call(self, prompt: str) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a strict JSON reranker for recommendation research."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
            "max_tokens": self.max_output_tokens,
            "response_format": {"type": "json_object"},
        }
        if self.provider == "deepseek":
            payload["thinking"] = {"type": "disabled"}
        request = urllib.request.Request(
            self.api_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                start = time.time()
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    body = json.loads(response.read().decode("utf-8"))
                body["_latency_seconds"] = time.time() - start
                return body
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, http.client.RemoteDisconnected, ConnectionResetError) as exc:
                last_error = exc
                time.sleep(min(2.0**attempt, 8.0))
        raise RuntimeError(f"Direct rerank request failed after retries: {last_error}")

    def rerank_one(self, prompt: str, n_candidates: int) -> Tuple[List[int], Dict[str, float], bool]:
        cache_path = self._cache_path(prompt)
        if cache_path.exists():
            record = json.loads(cache_path.read_text(encoding="utf-8"))
        else:
            record = self._call(prompt)
            cache_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        content = str(record["choices"][0]["message"]["content"])
        parse_failure = False
        try:
            parsed = json.loads(content)
            raw = parsed.get("ranking", parsed if isinstance(parsed, list) else [])
            order: List[int] = []
            seen = set()
            for value in raw:
                if isinstance(value, str):
                    match = re.search(r"\d+", value)
                    if not match:
                        continue
                    idx = int(match.group(0)) - 1
                else:
                    idx = int(value) - 1
                if 0 <= idx < n_candidates and idx not in seen:
                    order.append(idx)
                    seen.add(idx)
            order.extend(idx for idx in range(n_candidates) if idx not in seen)
        except Exception:
            parse_failure = True
            order = list(range(n_candidates))
        usage = record.get("usage", {})
        cost = {
            "input_tokens": float(usage.get("prompt_tokens", estimate_tokens(prompt))),
            "output_tokens": float(usage.get("completion_tokens", estimate_tokens(content))),
            "latency_seconds": float(record.get("_latency_seconds", 0.0)),
        }
        if len(order) < n_candidates:
            parse_failure = True
            order.extend(idx for idx in range(n_candidates) if idx not in set(order))
        return order[:n_candidates], cost, parse_failure


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


def write_report(outdir: Path, summary_frame: pd.DataFrame, reliability_frame: pd.DataFrame, cost_usd: float, parse_failures: int, args: argparse.Namespace) -> None:
    lines = [
        "# Amazon Books Direct DeepSeek Rerank Pilot",
        "",
        f"Dataset: Amazon Books subset. Sample source: `{args.candidate_run_dir}`.",
        f"Direct candidates: top-{args.direct_candidates}. Output: top-{args.topk}. History limit: {args.history_limit}.",
        f"Estimated direct rerank cost USD: {cost_usd:.6f}. Parse failures: {parse_failures}.",
        "",
        "## Performance",
        "",
        markdown_table(summary_frame),
        "",
        "## Reliability",
        "",
        markdown_table(reliability_frame),
        "",
    ]
    (outdir / "amazon_books_direct_rerank_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    ensure_dirs(args.outdir)
    train, _, _, _ = read_interaction_split(args.datadir)
    n_users = int(train["uid"].max()) + 1
    histories = build_ordered_histories(train, n_users)
    meta = load_metadata(args.datadir)
    users, targets, candidates = load_test_batch(args.candidate_run_dir, args.direct_candidates)

    reranker = DirectReranker(args)
    ranked = np.zeros((len(users), args.topk), dtype=np.int64)
    cost_rows: List[Dict[str, object]] = []
    parse_failures = 0
    for row, uid_np in enumerate(users):
        uid = int(uid_np)
        prompt = build_prompt(uid, candidates[row], histories, meta, args.history_limit)
        order, cost, failed = reranker.rerank_one(prompt, candidates.shape[1])
        ranked[row] = candidates[row, np.asarray(order[: args.topk], dtype=np.int64)]
        parse_failures += int(failed)
        cost_rows.append({"uid": uid, **cost, "parse_failure": int(failed)})
        if (row + 1) % 50 == 0 or row + 1 == len(users):
            print(f"direct rerank {row + 1}/{len(users)} users", flush=True)

    base_ranked = candidates[:, : args.topk]
    base_metrics = metrics_from_ranked(base_ranked, targets, args.topk)
    direct_metrics = metrics_from_ranked(ranked, targets, args.topk)
    summary_frame = pd.DataFrame([
        {"Method": "LightGCN", **summary(base_metrics)},
        {"Method": "DeepSeek Direct Rerank", **summary(direct_metrics)},
    ])
    reliability_frame = pd.DataFrame([
        {"Method": "LightGCN", "HarmRate": 0.0, "PositiveGainRate": 0.0, "MeanDeltaNDCG@20": 0.0, "PositiveGainSum": 0.0, "NegativeGainSum": 0.0, "GainHarmRatio": np.nan},
        {"Method": "DeepSeek Direct Rerank", **reliability(base_metrics, direct_metrics)},
    ])
    cost = pd.DataFrame(cost_rows)
    cost_usd = float((cost["input_tokens"].sum() / 1_000_000.0) * args.input_price_per_1m + (cost["output_tokens"].sum() / 1_000_000.0) * args.output_price_per_1m)
    summary_frame.to_csv(args.outdir / "direct_rerank_summary.csv", index=False)
    reliability_frame.to_csv(args.outdir / "direct_rerank_reliability.csv", index=False)
    cost.to_csv(args.outdir / "direct_rerank_cost_trace.csv", index=False)
    np.save(args.outdir / "direct_rerank_top20.npy", ranked)
    manifest = {
        "status": "completed",
        "dataset": "amazon_books_subset",
        "provider": args.provider,
        "model": args.model,
        "candidate_run_dir": str(args.candidate_run_dir),
        "users": int(len(users)),
        "direct_candidates": args.direct_candidates,
        "topk": args.topk,
        "history_limit": args.history_limit,
        "estimated_cost_usd": cost_usd,
        "parse_failures": int(parse_failures),
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (args.outdir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    write_report(args.outdir, summary_frame, reliability_frame, cost_usd, parse_failures, args)
    print(f"Done. Report: {args.outdir / 'amazon_books_direct_rerank_report.md'}", flush=True)


if __name__ == "__main__":
    main()

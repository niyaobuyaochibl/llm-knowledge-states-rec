#!/usr/bin/env python3
"""Amazon Books profile-before-ranking pilot over cached LightGCN candidates."""

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
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path("/root/temporal_popularity_pilot")
SRC = ROOT / "src"
SCRIPT_DIR = ROOT / "scripts"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SCRIPT_DIR))

from temporal_popularity.data import infer_shape, read_interaction_split  # noqa: E402
from run_egpr_profile_repair_pilot import build_ordered_histories  # noqa: E402
from run_llm_selective_invocation_pilot import ensure_dirs, estimate_tokens, zscore  # noqa: E402

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "based", "book", "books", "by", "for", "from", "in", "is", "it", "its",
    "like", "likes", "of", "on", "or", "prefers", "prefer", "reader", "reading", "that", "the", "to", "user", "with",
    "enjoy", "enjoys", "interested", "interest", "stories", "story", "novel", "novels",
}

CATEGORY_SYNONYMS: Dict[str, Tuple[str, ...]] = {
    "fiction": ("fiction", "novel", "novels", "literary", "story", "stories"),
    "romance": ("romance", "romantic", "love"),
    "mystery": ("mystery", "detective", "crime", "thriller", "suspense"),
    "fantasy": ("fantasy", "magic", "magical", "paranormal", "supernatural"),
    "science": ("science", "sci", "technology", "technical"),
    "history": ("history", "historical", "biography", "memoir"),
    "children": ("children", "childrens", "kids", "juvenile", "young", "teen"),
    "religion": ("religion", "religious", "christian", "spiritual", "spirituality"),
    "business": ("business", "management", "finance", "economics", "leadership"),
    "health": ("health", "fitness", "medical", "diet", "wellness"),
}


@dataclass
class BookMeta:
    titles: List[str]
    categories: List[List[str]]
    descriptions: List[str]
    tokens: List[Tuple[str, ...]]


@dataclass
class ClaimRecord:
    uid: int
    claim_id: int
    claim: str
    claim_type: str
    confidence: float
    support_count: int
    support_score: float
    support_weight: float
    status: str
    supporting_items: List[int]


@dataclass
class CandidateBatch:
    users: np.ndarray
    targets: np.ndarray
    candidates: np.ndarray
    scores: np.ndarray
    split_name: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", type=Path, default=ROOT / "data/amazon_books_subset")
    parser.add_argument("--candidate-run-dir", type=Path, default=ROOT / "results/egpr_profile_repair/amazon_books_seed42_lightgcn_1000")
    parser.add_argument("--outdir", type=Path, default=ROOT / "results/egpr_profile_repair/amazon_books_seed42_deepseek_1000_expressive5")
    parser.add_argument("--provider", choices=["deepseek", "openai"], default="deepseek")
    parser.add_argument("--model", default=None)
    parser.add_argument("--api-url", default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--profile-mode", choices=["api", "offline"], default="api")
    parser.add_argument("--api-timeout", type=float, default=90.0)
    parser.add_argument("--api-max-retries", type=int, default=12)
    parser.add_argument("--api-max-output-tokens", type=int, default=800)
    parser.add_argument("--claims-per-user", type=int, default=5)
    parser.add_argument("--history-limit", type=int, default=20)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--support-threshold", type=float, default=0.08)
    parser.add_argument("--lambda-grid", nargs="+", type=float, default=[0.05, 0.1, 0.2, 0.3, 0.5, 1.0])
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


def simple_tokens(text: str) -> Tuple[str, ...]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    expanded: List[str] = []
    for word in words:
        if len(word) <= 2 or word in STOPWORDS:
            continue
        expanded.append(word)
        for key, values in CATEGORY_SYNONYMS.items():
            if word in values:
                expanded.append(key)
    return tuple(sorted(set(expanded)))


def load_metadata(datadir: Path) -> BookMeta:
    rows: Dict[int, Dict[str, object]] = {}
    with (datadir / "item_metadata.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rows[int(row["iid"])] = row
    n_items = max(rows) + 1
    titles: List[str] = []
    categories: List[List[str]] = []
    descriptions: List[str] = []
    tokens: List[Tuple[str, ...]] = []
    for iid in range(n_items):
        row = rows.get(iid, {})
        title = str(row.get("title", "Unknown Book"))
        cats = [str(x) for x in row.get("categories", [])]
        desc = str(row.get("description", ""))
        titles.append(title)
        categories.append(cats)
        descriptions.append(desc)
        tokens.append(simple_tokens(" ".join([title, " ".join(cats), desc[:500]])))
    return BookMeta(titles=titles, categories=categories, descriptions=descriptions, tokens=tokens)


def load_batch(path: Path, split_name: str) -> CandidateBatch:
    data = np.load(path / f"candidates_lightgcn_{split_name}.npz")
    return CandidateBatch(
        users=data["users"].astype(np.int64),
        targets=data["targets"].astype(np.int64),
        candidates=data["candidates"].astype(np.int64),
        scores=data["scores"].astype(np.float32),
        split_name=split_name,
    )


def history_lines(uid: int, histories: Sequence[np.ndarray], meta: BookMeta, history_limit: int) -> List[str]:
    lines: List[str] = []
    for pos, iid_np in enumerate(histories[int(uid)][-history_limit:], start=1):
        iid = int(iid_np)
        cats = ", ".join(meta.categories[iid][:5]) if meta.categories[iid] else "Unknown"
        lines.append(f"{pos}. {meta.titles[iid]} | {cats}")
    return lines


def candidate_text(iid: int, meta: BookMeta) -> str:
    cats = ", ".join(meta.categories[iid][:5]) if meta.categories[iid] else "Unknown"
    return f"{meta.titles[iid]} | {cats}"


def build_profile_prompt(uid: int, histories: Sequence[np.ndarray], meta: BookMeta, history_limit: int, claims_per_user: int) -> str:
    lines = [
        f"Given this user's Amazon Books interaction history, generate {claims_per_user} concise preference claims.",
        "Generate a rich but evidence-aware reading profile with explicit and latent preferences.",
        "Include genre, topic, style, theme, audience, or reading intent when supported by the history.",
        "Do not mention exact book titles.",
        "Return valid JSON only with schema {\"claims\":[{\"claim\":\"...\",\"type\":\"genre/topic/style/theme/intent\",\"confidence\":0.0}]}",
        "",
        f"User: U{uid}",
        "User history:",
    ]
    lines.extend(history_lines(uid, histories, meta, history_limit) or ["None"])
    return "\n".join(lines)


class ProfileGenerator:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.cache_dir = args.cache_dir or (args.outdir / "profile_api_cache")
        ensure_dirs(self.cache_dir)
        env_name = "DEEPSEEK_API_KEY" if args.provider == "deepseek" else "OPENAI_API_KEY"
        self.api_key = os.environ.get(env_name)
        if args.profile_mode == "api" and not self.api_key:
            raise SystemExit(f"{env_name} is not set.")
        self.api_url = args.api_url or ("https://api.deepseek.com/chat/completions" if args.provider == "deepseek" else "https://api.openai.com/v1/chat/completions")

    def _cache_path(self, prompt: str) -> Path:
        key = hashlib.sha256((self.args.provider + "\n" + self.args.model + "\n" + prompt).encode("utf-8")).hexdigest()
        return self.cache_dir / f"{key}.json"

    def _call(self, prompt: str) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "model": self.args.model,
            "messages": [
                {"role": "system", "content": "You generate strict JSON user preference claims for recommendation research."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
            "max_tokens": self.args.api_max_output_tokens,
            "response_format": {"type": "json_object"},
        }
        if self.args.provider == "deepseek":
            payload["thinking"] = {"type": "disabled"}
        request = urllib.request.Request(
            self.api_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        last_error: Optional[Exception] = None
        for attempt in range(self.args.api_max_retries):
            try:
                start = time.time()
                with urllib.request.urlopen(request, timeout=self.args.api_timeout) as response:
                    body = json.loads(response.read().decode("utf-8"))
                body["_latency_seconds"] = time.time() - start
                return body
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, http.client.RemoteDisconnected, ConnectionResetError) as exc:
                last_error = exc
                time.sleep(min(2.0**attempt, 8.0))
        raise RuntimeError(f"Profile request failed after retries: {last_error}")

    def generate(self, uid: int, prompt: str) -> Tuple[List[Dict[str, object]], Dict[str, float]]:
        if self.args.profile_mode == "offline":
            return [{"claim": "reads books related to recent history topics", "type": "topic", "confidence": 0.1}], {"input_tokens": 0.0, "output_tokens": 0.0, "latency_seconds": 0.0}
        cache_path = self._cache_path(prompt)
        if cache_path.exists():
            record = json.loads(cache_path.read_text(encoding="utf-8"))
        else:
            record = self._call(prompt)
            cache_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        content = str(record["choices"][0]["message"]["content"])
        parsed = json.loads(content)
        raw_claims = parsed.get("claims", parsed if isinstance(parsed, list) else [])
        claims: List[Dict[str, object]] = []
        for item in raw_claims[: self.args.claims_per_user]:
            if isinstance(item, dict) and str(item.get("claim", "")).strip():
                claims.append({
                    "claim": str(item.get("claim", "")).strip(),
                    "type": str(item.get("type", "preference")).strip() or "preference",
                    "confidence": float(item.get("confidence", 0.0) or 0.0),
                })
        usage = record.get("usage", {})
        cost = {
            "input_tokens": float(usage.get("prompt_tokens", estimate_tokens(prompt))),
            "output_tokens": float(usage.get("completion_tokens", estimate_tokens(content))),
            "latency_seconds": float(record.get("_latency_seconds", 0.0)),
        }
        return claims, cost


def token_overlap_score(claim: str, item_tokens: Sequence[str]) -> float:
    claim_tokens = set(simple_tokens(claim))
    if not claim_tokens or not item_tokens:
        return 0.0
    item_set = set(item_tokens)
    return len(claim_tokens & item_set) / math.sqrt(len(claim_tokens) * len(item_set))


def support_weight(score: float, count: int) -> float:
    return float(1.0 / (1.0 + math.exp(-8.0 * (score + min(count, 5) / 5.0 * 0.2 - 0.18))))


def score_claim(uid: int, claim_id: int, claim: Mapping[str, object], histories: Sequence[np.ndarray], meta: BookMeta, history_limit: int, threshold: float) -> ClaimRecord:
    scores: List[Tuple[int, float]] = []
    for iid_np in histories[int(uid)][-history_limit:]:
        iid = int(iid_np)
        score = token_overlap_score(str(claim["claim"]), meta.tokens[iid])
        if score >= threshold:
            scores.append((iid, score))
    support_count = len(scores)
    best = max((score for _, score in scores), default=0.0)
    avg = float(np.mean([score for _, score in scores])) if scores else 0.0
    support_score = float(0.7 * best + 0.3 * avg)
    status = "supported" if support_count >= 2 else ("weakly_supported" if support_count == 1 else "unsupported")
    return ClaimRecord(
        uid=int(uid),
        claim_id=int(claim_id),
        claim=str(claim["claim"]),
        claim_type=str(claim.get("type", "preference")),
        confidence=float(claim.get("confidence", 0.0) or 0.0),
        support_count=support_count,
        support_score=support_score,
        support_weight=support_weight(support_score, support_count),
        status=status,
        supporting_items=[iid for iid, _ in scores],
    )


def score_all(profiles: Mapping[int, List[Dict[str, object]]], histories: Sequence[np.ndarray], meta: BookMeta, history_limit: int, threshold: float) -> Dict[int, List[ClaimRecord]]:
    out: Dict[int, List[ClaimRecord]] = {}
    for uid, claims in profiles.items():
        out[int(uid)] = [score_claim(int(uid), idx, claim, histories, meta, history_limit, threshold) for idx, claim in enumerate(claims)]
    return out


def profile_scores(batch: CandidateBatch, records: Mapping[int, List[ClaimRecord]], meta: BookMeta, method: str) -> np.ndarray:
    scores = np.zeros_like(batch.scores, dtype=np.float32)
    for row, uid_np in enumerate(batch.users):
        weighted: List[Tuple[ClaimRecord, float]] = []
        for record in records.get(int(uid_np), []):
            if method == "raw":
                weight = 1.0
            elif method == "remove":
                weight = 0.0 if record.status == "unsupported" else 1.0
            elif method == "weighted":
                weight = record.support_weight
            else:
                raise ValueError(method)
            if weight > 0.0:
                weighted.append((record, weight))
        if not weighted:
            continue
        denom = sum(weight for _, weight in weighted)
        for pos, iid_np in enumerate(batch.candidates[row]):
            iid = int(iid_np)
            total = sum(weight * token_overlap_score(record.claim, meta.tokens[iid]) for record, weight in weighted)
            scores[row, pos] = float(total / max(denom, 1e-8))
    return scores


def ranked_from_scores(batch: CandidateBatch, pscores: np.ndarray, lam: float, topk: int) -> np.ndarray:
    ranked = np.zeros((len(batch.users), topk), dtype=np.int64)
    for row in range(len(batch.users)):
        final = zscore(batch.scores[row]) + lam * zscore(pscores[row])
        order = np.lexsort((batch.candidates[row], -final))[:topk]
        ranked[row] = batch.candidates[row, order]
    return ranked


def metrics_from_ranked(ranked: np.ndarray, targets: np.ndarray, topk: int) -> pd.DataFrame:
    rows = []
    for row, target_np in enumerate(targets):
        pos = np.flatnonzero(ranked[row, :topk] == int(target_np))
        hit = len(pos) > 0
        rows.append({"NDCG@20": 1.0 / math.log2(int(pos[0]) + 2) if hit else 0.0, "Recall@20": float(hit), "HitRate@20": float(hit)})
    return pd.DataFrame(rows)


def summary(metrics: pd.DataFrame) -> Dict[str, float]:
    return {col: float(metrics[col].mean()) for col in ["NDCG@20", "Recall@20", "HitRate@20"]}


def reliability(base: pd.DataFrame, method: pd.DataFrame) -> Dict[str, float]:
    delta = method["NDCG@20"].to_numpy(np.float64) - base["NDCG@20"].to_numpy(np.float64)
    pos = delta[delta > 0]
    neg = delta[delta < 0]
    return {
        "HarmRate": float(np.mean(delta < 0)),
        "PositiveGainRate": float(np.mean(delta > 0)),
        "MeanDeltaNDCG@20": float(delta.mean()),
        "PositiveGainSum": float(pos.sum()) if len(pos) else 0.0,
        "NegativeGainSum": float(neg.sum()) if len(neg) else 0.0,
        "GainHarmRatio": float(pos.sum() / abs(neg.sum())) if len(neg) and abs(float(neg.sum())) > 0 else np.inf,
    }


def faithfulness(records: Mapping[int, List[ClaimRecord]], histories: Sequence[np.ndarray], meta: BookMeta, history_limit: int) -> pd.DataFrame:
    rows = []
    for label, mode in [("Raw Profile", "raw"), ("Remove Repair", "remove"), ("Evidence-Weighted Repair", "weighted")]:
        claims = unsupported = 0
        weighted_unsupported = total_weight = 0.0
        drift_values = []
        coverage_values = []
        for uid, recs in records.items():
            retained = []
            for rec in recs:
                if mode == "remove" and rec.status == "unsupported":
                    continue
                weight = rec.support_weight if mode == "weighted" else 1.0
                retained.append((rec, weight))
                claims += 1
                total_weight += weight
                if rec.status == "unsupported":
                    unsupported += 1
                    weighted_unsupported += weight
            hist = histories[int(uid)][-history_limit:]
            covered = set()
            for rec, _ in retained:
                covered.update(rec.supporting_items)
            coverage_values.append(len(covered.intersection(set(int(x) for x in hist))) / len(hist) if len(hist) else 0.0)
            profile_vec: Dict[str, float] = {}
            hist_vec: Dict[str, float] = {}
            for rec, weight in retained:
                for tok in simple_tokens(rec.claim):
                    profile_vec[tok] = profile_vec.get(tok, 0.0) + weight
            for iid_np in hist:
                for tok in meta.tokens[int(iid_np)]:
                    hist_vec[tok] = hist_vec.get(tok, 0.0) + 1.0
            drift_values.append(1.0 - cosine_sparse(profile_vec, hist_vec))
        rows.append({
            "Method": label,
            "Claims": claims,
            "UCR": unsupported / claims if claims else 0.0,
            "WeightedUCR": weighted_unsupported / total_weight if total_weight else 0.0,
            "EvidenceCoverage": float(np.mean(coverage_values)) if coverage_values else 0.0,
            "ProfileDriftScore": float(np.mean(drift_values)) if drift_values else 1.0,
        })
    return pd.DataFrame(rows)


def cosine_sparse(a: Mapping[str, float], b: Mapping[str, float]) -> float:
    if not a or not b:
        return 0.0
    common = set(a) & set(b)
    num = sum(a[k] * b[k] for k in common)
    den = math.sqrt(sum(v * v for v in a.values())) * math.sqrt(sum(v * v for v in b.values()))
    return float(num / den) if den > 0 else 0.0


def write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def markdown_table(frame: pd.DataFrame) -> str:
    display = frame.copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(lambda x: "inf" if np.isinf(x) else f"{x:.6f}")
        else:
            display[col] = display[col].astype(str)
    lines = ["| " + " | ".join(display.columns) + " |", "| " + " | ".join(["---"] * len(display.columns)) + " |"]
    for row in display.values.tolist():
        lines.append("| " + " | ".join(map(str, row)) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    ensure_dirs(args.outdir)
    train, _, _, _ = read_interaction_split(args.datadir)
    n_users = int(train["uid"].max()) + 1
    histories = build_ordered_histories(train, n_users)
    meta = load_metadata(args.datadir)
    val_batch = load_batch(args.candidate_run_dir, "val")
    test_batch = load_batch(args.candidate_run_dir, "test")
    profile_users = sorted(set(val_batch.users.astype(int).tolist() + test_batch.users.astype(int).tolist()))

    generator = ProfileGenerator(args)
    profiles: Dict[int, List[Dict[str, object]]] = {}
    cost_rows = []
    for idx, uid in enumerate(profile_users, start=1):
        prompt = build_profile_prompt(uid, histories, meta, args.history_limit, args.claims_per_user)
        claims, cost = generator.generate(uid, prompt)
        profiles[int(uid)] = claims
        cost_rows.append({"uid": int(uid), **cost})
        if idx % 50 == 0 or idx == len(profile_users):
            print(f"profile generation {idx}/{len(profile_users)} users", flush=True)
    write_jsonl(args.outdir / "raw_profiles.jsonl", [{"uid": uid, "claims": claims} for uid, claims in profiles.items()])
    pd.DataFrame(cost_rows).to_csv(args.outdir / "profile_cost_trace.csv", index=False)

    records = score_all(profiles, histories, meta, args.history_limit, args.support_threshold)
    write_jsonl(args.outdir / "claim_support.jsonl", [rec.__dict__ for recs in records.values() for rec in recs])

    faith = faithfulness(records, histories, meta, args.history_limit)
    faith.to_csv(args.outdir / "table1_profile_faithfulness.csv", index=False)
    lambda_rows = []
    performance_rows = [{"Method": "LightGCN", "SelectedLambda": 0.0, **summary(metrics_from_ranked(test_batch.candidates[:, :args.topk], test_batch.targets, args.topk))}]
    reliability_rows = [{"Method": "LightGCN", "HarmRate": 0.0, "PositiveGainRate": 0.0, "MeanDeltaNDCG@20": 0.0, "PositiveGainSum": 0.0, "NegativeGainSum": 0.0, "GainHarmRatio": np.nan}]
    per_user = {"lightgcn": metrics_from_ranked(test_batch.candidates[:, :args.topk], test_batch.targets, args.topk)}
    base_metrics = per_user["lightgcn"]
    for label, method_key, file_key in [("LightGCN + Raw Profile", "raw", "raw"), ("LightGCN + Remove Repair", "remove", "remove"), ("LightGCN + Evidence-Weighted Repair", "weighted", "weighted")]:
        val_scores = profile_scores(val_batch, records, meta, method_key)
        test_scores = profile_scores(test_batch, records, meta, method_key)
        best_lam = args.lambda_grid[0]
        best_ndcg = -1.0
        for lam in args.lambda_grid:
            ranked_val = ranked_from_scores(val_batch, val_scores, lam, args.topk)
            val_metrics = metrics_from_ranked(ranked_val, val_batch.targets, args.topk)
            row = {"Method": label, "Lambda": lam, **summary(val_metrics)}
            lambda_rows.append(row)
            if row["NDCG@20"] > best_ndcg:
                best_ndcg = row["NDCG@20"]
                best_lam = lam
        ranked_test = ranked_from_scores(test_batch, test_scores, best_lam, args.topk)
        test_metrics = metrics_from_ranked(ranked_test, test_batch.targets, args.topk)
        per_user[file_key] = test_metrics
        performance_rows.append({"Method": label, "SelectedLambda": best_lam, **summary(test_metrics)})
        reliability_rows.append({"Method": label, **reliability(base_metrics, test_metrics)})
        np.save(args.outdir / f"ranked_{file_key}_top20.npy", ranked_test)
    perf = pd.DataFrame(performance_rows)
    rel = pd.DataFrame(reliability_rows)
    lambdas = pd.DataFrame(lambda_rows)
    perf.to_csv(args.outdir / "table2_recommendation_performance.csv", index=False)
    rel.to_csv(args.outdir / "table3_reliability.csv", index=False)
    lambdas.to_csv(args.outdir / "table4_lambda_validation.csv", index=False)
    for key, frame in per_user.items():
        out = frame.copy()
        out.insert(0, "target", test_batch.targets)
        out.insert(0, "uid", test_batch.users)
        out.to_csv(args.outdir / f"per_user_{key}.csv", index=False)

    cost_df = pd.DataFrame(cost_rows)
    cost_usd = float((cost_df["input_tokens"].sum() / 1_000_000.0) * args.input_price_per_1m + (cost_df["output_tokens"].sum() / 1_000_000.0) * args.output_price_per_1m) if len(cost_df) else 0.0
    manifest = {
        "status": "completed",
        "dataset": "amazon_books_subset",
        "provider": args.provider,
        "model": args.model,
        "candidate_run_dir": str(args.candidate_run_dir),
        "profile_users": len(profile_users),
        "test_users": int(len(test_batch.users)),
        "topk": args.topk,
        "claims_per_user": args.claims_per_user,
        "estimated_profile_cost_usd": cost_usd,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (args.outdir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Amazon Books Profile Rerank Pilot",
        "",
        f"Dataset: Amazon Books subset. Baseline: LightGCN. Candidate set: top-{test_batch.candidates.shape[1]}. Output: top-{args.topk}.",
        f"Users per split: {len(test_batch.users)}. History limit: {args.history_limit}. Claims per user: {args.claims_per_user}.",
        f"Estimated profile generation cost USD: {cost_usd:.6f}.",
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
        "## Artifacts",
        "",
        "- `raw_profiles.jsonl`",
        "- `claim_support.jsonl`",
        "- `table1_profile_faithfulness.csv`",
        "- `table2_recommendation_performance.csv`",
        "- `table3_reliability.csv`",
        "- `table4_lambda_validation.csv`",
        "- `profile_cost_trace.csv`",
        "- `run_manifest.json`",
    ]
    (args.outdir / "amazon_books_profile_rerank_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Done. Report: {args.outdir / 'amazon_books_profile_rerank_report.md'}", flush=True)


if __name__ == "__main__":
    main()

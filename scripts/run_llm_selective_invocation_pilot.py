#!/usr/bin/env python3
"""ML-1M selective LLM invocation pilot.

The script has two execution modes:

* proxy: deterministic metadata-aware reranker for offline plumbing checks.
  This is not an LLM result and should not be reported as one.
* openai: calls the OpenAI Chat Completions API through urllib and caches every
  response. API mode requires OPENAI_API_KEY or DEEPSEEK_API_KEY, depending on provider.

The goal is to make the proposed pilot reproducible without disturbing the
existing temporal-popularity experiment outputs.
"""

from __future__ import annotations

import argparse
import hashlib
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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

ROOT = Path("/root/temporal_popularity_pilot")
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from temporal_popularity.data import (  # noqa: E402
    activity_controlled_user_groups,
    build_exclude_lists,
    build_user_histories,
    infer_shape,
    read_interaction_split,
    set_seed,
)
from temporal_popularity.eval import topk_indices  # noqa: E402
from temporal_popularity.model import LightGCN, build_norm_adj  # noqa: E402
from temporal_popularity.popularity import assign_buckets, popularity_percentiles, static_popularity  # noqa: E402
from temporal_popularity.sequential import (  # noqa: E402
    SASRec,
    build_ordered_user_sequences,
    sequence_rows_for_users,
)


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "for",
    "from",
    "in",
    "is",
    "of",
    "on",
    "or",
    "part",
    "the",
    "to",
    "with",
}


@dataclass
class ItemMetadata:
    titles: List[str]
    genres: List[Tuple[str, ...]]
    title_tokens: List[Tuple[str, ...]]
    years: np.ndarray
    text_lengths: np.ndarray


@dataclass
class CandidateBatch:
    users: np.ndarray
    targets: np.ndarray
    candidates: np.ndarray
    scores: np.ndarray
    histories: List[np.ndarray]
    split_name: str


@dataclass
class RerankBatch:
    ranked_topk: np.ndarray
    input_tokens: np.ndarray
    output_tokens: np.ndarray
    latency_seconds: np.ndarray
    parse_failures: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", type=Path, default=ROOT / "data/ml1m")
    parser.add_argument(
        "--movies-path",
        type=Path,
        default=Path("/root/autodl-tmp/projects_archive/MGPrompt-CDR/data/raw/ml-1m/movies.dat"),
    )
    parser.add_argument("--config", type=Path, default=ROOT / "results/formal/ml1m/seed42/config.json")
    parser.add_argument("--lightgcn-ckpt", type=Path, default=ROOT / "results/formal/ml1m/seed42/lightgcn.pt")
    parser.add_argument(
        "--sasrec-ckpt",
        type=Path,
        default=ROOT / "results/formal/backbone_robustness/ml1m/seed42/sasrec/sasrec.pt",
    )
    parser.add_argument("--outdir", type=Path, default=ROOT / "results/llm_selective/ml1m_seed42_proxy")
    parser.add_argument("--figdir", type=Path, default=ROOT / "figures/llm_selective/ml1m_seed42_proxy")
    parser.add_argument("--baselines", nargs="+", default=["pop", "lightgcn", "sasrec"])
    parser.add_argument("--mode", choices=["proxy", "api"], default="proxy")
    parser.add_argument("--provider", choices=["openai", "deepseek"], default="openai")
    parser.add_argument("--model", default=None)
    parser.add_argument("--api-url", default=None)
    parser.add_argument("--top-candidates", type=int, default=50)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--history-limit", type=int, default=10)
    parser.add_argument("--budgets", nargs="+", type=float, default=[0.10, 0.20, 0.30, 0.50, 1.00])
    parser.add_argument("--max-users", type=int, default=0, help="0 means use all users in each split.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--random-trials", type=int, default=20)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--openai-timeout", type=float, default=60.0)
    parser.add_argument("--openai-max-retries", type=int, default=3)
    parser.add_argument("--openai-max-output-tokens", type=int, default=350)
    parser.add_argument("--proxy-latency-seconds", type=float, default=0.0)
    parser.add_argument(
        "--input-price-per-1m",
        type=float,
        default=None,
        help="If omitted, provider/model defaults are used for cost estimates.",
    )
    parser.add_argument(
        "--output-price-per-1m",
        type=float,
        default=None,
        help="If omitted, provider/model defaults are used for cost estimates.",
    )
    parser.add_argument(
        "--allow-leaky-gate-feature",
        action="store_true",
        help="Include ground-truth target rank in learned gate features. Off by default because it is not deployable.",
    )
    parser.add_argument("--force-api-diagnostics", action="store_true")
    args = parser.parse_args()
    if args.model is None:
        args.model = "deepseek-v4-flash" if args.provider == "deepseek" else "gpt-4.1-mini"
    if args.input_price_per_1m is None:
        args.input_price_per_1m = 0.14 if args.provider == "deepseek" else 0.40
    if args.output_price_per_1m is None:
        args.output_price_per_1m = 0.28 if args.provider == "deepseek" else 1.60
    return args


def ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def device_from_arg(arg: str) -> torch.device:
    if arg == "cpu":
        return torch.device("cpu")
    if arg == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def zscore(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    std = float(arr.std())
    if std < 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - float(arr.mean())) / std).astype(np.float32)


def softmax_entropy(scores: np.ndarray) -> float:
    vals = np.asarray(scores, dtype=np.float64)
    finite = np.isfinite(vals)
    if not finite.any():
        return 0.0
    vals = np.where(finite, vals, float(vals[finite].min()) - 1.0)
    vals = vals - float(vals.max())
    exp = np.exp(vals)
    denom = float(exp.sum())
    if denom <= 0.0:
        return 0.0
    prob = exp / denom
    entropy = -float(np.sum(prob * np.log(prob + 1e-12)))
    return entropy / math.log(len(scores)) if len(scores) > 1 else 0.0


def tokenize_title(title: str) -> Tuple[str, ...]:
    tokens = []
    for token in re.findall(r"[A-Za-z0-9]+", title.lower()):
        if len(token) > 1 and token not in STOPWORDS and not token.isdigit():
            tokens.append(token)
    return tuple(sorted(set(tokens)))


def parse_movie_year(title: str) -> float:
    matches = re.findall(r"\((\d{4})\)", title)
    if not matches:
        return np.nan
    return float(matches[-1])


def read_item_metadata(movies_path: Path, mappings_path: Path, n_items: int) -> ItemMetadata:
    mappings = json.loads(mappings_path.read_text(encoding="utf-8"))
    item2id = {str(key): int(value) for key, value in mappings["item2id"].items()}

    titles = [f"Movie_{iid}" for iid in range(n_items)]
    genres: List[Tuple[str, ...]] = [tuple() for _ in range(n_items)]
    title_tokens: List[Tuple[str, ...]] = [tuple() for _ in range(n_items)]
    years = np.full(n_items, np.nan, dtype=np.float32)

    if movies_path.exists():
        with movies_path.open("r", encoding="latin-1") as handle:
            for line in handle:
                parts = line.rstrip("\n").split("::")
                if len(parts) != 3:
                    continue
                raw_iid, title, genre_text = parts
                if raw_iid not in item2id:
                    continue
                iid = item2id[raw_iid]
                titles[iid] = title
                genres[iid] = tuple(g for g in genre_text.split("|") if g)
                title_tokens[iid] = tokenize_title(title)
                years[iid] = parse_movie_year(title)
    else:
        print(f"WARNING: movie metadata not found: {movies_path}", flush=True)

    text_lengths = np.asarray(
        [len(title_tokens[iid]) + sum(len(g.split()) for g in genres[iid]) for iid in range(n_items)],
        dtype=np.float32,
    )
    return ItemMetadata(titles=titles, genres=genres, title_tokens=title_tokens, years=years, text_lengths=text_lengths)


def sample_eval_frame(frame: pd.DataFrame, max_users: int, seed: int) -> pd.DataFrame:
    ordered = frame.sort_values("uid", kind="mergesort").reset_index(drop=True)
    if max_users <= 0 or max_users >= len(ordered):
        return ordered
    rng = np.random.default_rng(seed)
    chosen = np.sort(rng.choice(len(ordered), size=max_users, replace=False))
    return ordered.iloc[chosen].reset_index(drop=True)


def baseline_metrics_from_ranked(ranked_topk: np.ndarray, targets: np.ndarray) -> pd.DataFrame:
    rows = []
    for row_idx, target_np in enumerate(targets):
        target = int(target_np)
        positions = np.flatnonzero(ranked_topk[row_idx] == target)
        hit = len(positions) > 0
        ndcg = 1.0 / math.log2(int(positions[0]) + 2) if hit else 0.0
        rows.append({"NDCG@20": ndcg, "Recall@20": float(hit), "HitRate@20": float(hit)})
    return pd.DataFrame(rows)


def estimate_tokens(text: str) -> int:
    word_count = len(re.findall(r"\S+", text))
    return max(1, int(math.ceil(word_count * 1.35)))


def item_prompt_line(code: str, iid: int, meta: ItemMetadata, title_mask: bool) -> str:
    title = f"Movie_{iid}" if title_mask else meta.titles[iid]
    genre_text = "|".join(meta.genres[iid]) if meta.genres[iid] else "Unknown"
    year = "" if np.isnan(meta.years[iid]) or title_mask else f", year={int(meta.years[iid])}"
    return f"{code}: {title} [genres={genre_text}{year}]"


def build_prompt(
    uid: int,
    history: np.ndarray,
    candidates: np.ndarray,
    meta: ItemMetadata,
    history_limit: int,
    title_mask: bool,
) -> str:
    recent = history[-history_limit:] if len(history) else np.asarray([], dtype=np.int64)
    lines = [
        "Task: rerank candidate movies for one MovieLens user.",
        "Use the recent history and item metadata only. Return JSON only.",
        'Output schema: {"ranking":["C01","C02",...]} with exactly 20 unique candidate IDs.',
        f"User: U{uid}",
        "Recent history:",
    ]
    if len(recent) == 0:
        lines.append("None")
    else:
        for pos, iid_np in enumerate(recent, start=1):
            lines.append(item_prompt_line(f"H{pos:02d}", int(iid_np), meta, title_mask))
    lines.append("Candidates:")
    for pos, iid_np in enumerate(candidates, start=1):
        lines.append(item_prompt_line(f"C{pos:02d}", int(iid_np), meta, title_mask))
    return "\n".join(lines)


def estimated_output_tokens(topk: int) -> int:
    sample_codes = [f"C{i:02d}" for i in range(1, topk + 1)]
    return estimate_tokens(json.dumps({"ranking": sample_codes}, separators=(",", ":")))


def make_prompt_token_arrays(
    users: np.ndarray,
    histories: Sequence[np.ndarray],
    candidates: np.ndarray,
    meta: ItemMetadata,
    history_limit: int,
    title_mask: bool,
    topk: int,
) -> Tuple[np.ndarray, np.ndarray]:
    input_tokens = np.zeros(len(users), dtype=np.int32)
    output_tokens = np.zeros(len(users), dtype=np.int32)
    out_tok = estimated_output_tokens(topk)
    for row, uid_np in enumerate(users):
        prompt = build_prompt(int(uid_np), histories[int(uid_np)], candidates[row], meta, history_limit, title_mask)
        input_tokens[row] = estimate_tokens(prompt)
        output_tokens[row] = out_tok
    return input_tokens, output_tokens


def load_lightgcn_embeddings(
    config_path: Path,
    checkpoint_path: Path,
    train: pd.DataFrame,
    n_users: int,
    n_items: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"LightGCN checkpoint not found: {checkpoint_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    model_cfg = config["backbone"]
    model = LightGCN(n_users, n_items, int(model_cfg["embedding_dim"]), int(model_cfg["layers"])).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    norm_adj = build_norm_adj(train, n_users, n_items, device)
    model.eval()
    with torch.no_grad():
        user_emb_t, item_emb_t = model.propagate(norm_adj)
    return user_emb_t.detach().cpu().numpy().astype(np.float32), item_emb_t.detach().cpu().numpy().astype(np.float32)


def load_sasrec_model(checkpoint_path: Path, n_items: int, device: torch.device) -> SASRec:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"SASRec checkpoint not found: {checkpoint_path}")
    model = SASRec(n_items=n_items, max_len=50, embedding_dim=64, layers=2, heads=2, dropout=0.2).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def top_candidates_for_scores(
    score_vector: np.ndarray,
    excluded: np.ndarray,
    topn: int,
) -> Tuple[np.ndarray, np.ndarray]:
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


def generate_pop_candidates(
    eval_frame: pd.DataFrame,
    histories: Sequence[np.ndarray],
    exclude_lists: Sequence[np.ndarray],
    static_pop: np.ndarray,
    topn: int,
    split_name: str,
) -> CandidateBatch:
    users = eval_frame["uid"].to_numpy(np.int64)
    targets = eval_frame["iid"].to_numpy(np.int64)
    candidates = np.zeros((len(users), topn), dtype=np.int64)
    scores = np.zeros((len(users), topn), dtype=np.float32)
    base_scores = static_pop.astype(np.float32)
    for row, uid_np in enumerate(users):
        uid = int(uid_np)
        candidates[row], scores[row] = top_candidates_for_scores(base_scores, exclude_lists[uid], topn)
    return CandidateBatch(users, targets, candidates, scores, list(histories), split_name)


def generate_lightgcn_candidates(
    eval_frame: pd.DataFrame,
    histories: Sequence[np.ndarray],
    exclude_lists: Sequence[np.ndarray],
    user_emb: np.ndarray,
    item_emb: np.ndarray,
    topn: int,
    batch_size: int,
    split_name: str,
) -> CandidateBatch:
    users = eval_frame["uid"].to_numpy(np.int64)
    targets = eval_frame["iid"].to_numpy(np.int64)
    candidates = np.zeros((len(users), topn), dtype=np.int64)
    scores = np.zeros((len(users), topn), dtype=np.float32)
    item_emb_t = item_emb.T.astype(np.float32)
    for start in range(0, len(users), batch_size):
        end = min(start + batch_size, len(users))
        batch_users = users[start:end]
        score_batch = user_emb[batch_users].astype(np.float32) @ item_emb_t
        for local, uid_np in enumerate(batch_users):
            uid = int(uid_np)
            candidates[start + local], scores[start + local] = top_candidates_for_scores(
                score_batch[local], exclude_lists[uid], topn
            )
    return CandidateBatch(users, targets, candidates, scores, list(histories), split_name)


def generate_sasrec_candidates(
    eval_frame: pd.DataFrame,
    histories: Sequence[np.ndarray],
    exclude_lists: Sequence[np.ndarray],
    sequences: Sequence[np.ndarray],
    model: SASRec,
    n_items: int,
    topn: int,
    batch_size: int,
    device: torch.device,
    split_name: str,
) -> CandidateBatch:
    users = eval_frame["uid"].to_numpy(np.int64)
    targets = eval_frame["iid"].to_numpy(np.int64)
    candidates = np.zeros((len(users), topn), dtype=np.int64)
    scores = np.zeros((len(users), topn), dtype=np.float32)
    model.eval()
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
    return CandidateBatch(users, targets, candidates, scores, list(histories), split_name)


def weighted_profile(items: np.ndarray, meta: ItemMetadata, history_limit: int) -> Tuple[Dict[str, float], Dict[str, float], float]:
    recent = items[-history_limit:] if len(items) else np.asarray([], dtype=np.int64)
    genre_profile: Dict[str, float] = {}
    token_profile: Dict[str, float] = {}
    year_values: List[float] = []
    if len(recent) == 0:
        return genre_profile, token_profile, np.nan
    denom = float(len(recent))
    for pos, iid_np in enumerate(recent, start=1):
        iid = int(iid_np)
        weight = pos / denom
        for genre in meta.genres[iid]:
            genre_profile[genre] = genre_profile.get(genre, 0.0) + weight
        for token in meta.title_tokens[iid]:
            token_profile[token] = token_profile.get(token, 0.0) + weight
        if not np.isnan(meta.years[iid]):
            year_values.append(float(meta.years[iid]))
    median_year = float(np.median(year_values)) if year_values else np.nan
    return genre_profile, token_profile, median_year


def sparse_cosine(profile: Mapping[str, float], keys: Iterable[str]) -> float:
    key_list = list(keys)
    if not profile or not key_list:
        return 0.0
    numerator = sum(float(profile.get(key, 0.0)) for key in key_list)
    left = math.sqrt(sum(float(value) * float(value) for value in profile.values()))
    right = math.sqrt(float(len(key_list)))
    if left <= 0.0 or right <= 0.0:
        return 0.0
    return numerator / (left * right)


def proxy_rank_one(
    uid: int,
    history: np.ndarray,
    candidate_items: np.ndarray,
    baseline_scores: np.ndarray,
    meta: ItemMetadata,
    static_pop: np.ndarray,
    history_limit: int,
    topk: int,
    title_mask: bool,
) -> np.ndarray:
    genre_profile, token_profile, median_year = weighted_profile(history, meta, history_limit)
    semantic = np.zeros(len(candidate_items), dtype=np.float32)
    for pos, iid_np in enumerate(candidate_items):
        iid = int(iid_np)
        genre_sim = sparse_cosine(genre_profile, meta.genres[iid])
        token_sim = 0.0 if title_mask else sparse_cosine(token_profile, meta.title_tokens[iid])
        year_sim = 0.0
        if not title_mask and not np.isnan(median_year) and not np.isnan(meta.years[iid]):
            year_sim = 1.0 / (1.0 + abs(float(meta.years[iid]) - median_year) / 10.0)
        if title_mask:
            semantic[pos] = float(genre_sim)
        else:
            semantic[pos] = float(0.50 * genre_sim + 0.35 * token_sim + 0.15 * year_sim)
    combined = 0.62 * zscore(baseline_scores) + 0.33 * zscore(semantic) + 0.05 * zscore(np.log1p(static_pop[candidate_items]))
    order = np.argsort(-combined, kind="mergesort")
    return candidate_items[order[:topk]]


def build_perturbed_histories(
    users: np.ndarray,
    base_histories: Sequence[np.ndarray],
    static_pop: np.ndarray,
    variant: str,
    history_limit: int,
    seed: int,
) -> Dict[int, np.ndarray]:
    if variant == "original" or variant == "title_mask":
        return {int(uid): base_histories[int(uid)] for uid in users}
    rng = np.random.default_rng(seed + 1009)
    user_list = np.asarray([int(uid) for uid in users], dtype=np.int64)
    popular_items = np.argsort(-static_pop, kind="mergesort")[:history_limit].astype(np.int64)
    output: Dict[int, np.ndarray] = {}
    for uid_np in users:
        uid = int(uid_np)
        hist = np.asarray(base_histories[uid], dtype=np.int64)
        if variant == "shuffled_history":
            shuffled = hist.copy()
            if len(shuffled):
                local_rng = np.random.default_rng(seed + uid)
                local_rng.shuffle(shuffled)
            output[uid] = shuffled
        elif variant == "random_history":
            other = int(rng.choice(user_list))
            if len(user_list) > 1:
                while other == uid:
                    other = int(rng.choice(user_list))
            output[uid] = np.asarray(base_histories[other], dtype=np.int64)
        elif variant == "popular_history":
            output[uid] = popular_items.copy()
        else:
            raise ValueError(f"Unknown history variant: {variant}")
    return output


def proxy_rerank_batch(
    batch: CandidateBatch,
    meta: ItemMetadata,
    static_pop: np.ndarray,
    history_limit: int,
    topk: int,
    variant: str,
    seed: int,
    proxy_latency_seconds: float,
) -> RerankBatch:
    title_mask = variant == "title_mask"
    histories = build_perturbed_histories(batch.users, batch.histories, static_pop, variant, history_limit, seed)
    ranked = np.zeros((len(batch.users), topk), dtype=np.int64)
    input_tokens = np.zeros(len(batch.users), dtype=np.int32)
    output_tokens = np.zeros(len(batch.users), dtype=np.int32)
    latency = np.full(len(batch.users), float(proxy_latency_seconds), dtype=np.float32)
    out_tok = estimated_output_tokens(topk)
    for row, uid_np in enumerate(batch.users):
        uid = int(uid_np)
        ranked[row] = proxy_rank_one(
            uid,
            histories[uid],
            batch.candidates[row],
            batch.scores[row],
            meta,
            static_pop,
            history_limit,
            topk,
            title_mask,
        )
        prompt = build_prompt(uid, histories[uid], batch.candidates[row], meta, history_limit, title_mask)
        input_tokens[row] = estimate_tokens(prompt)
        output_tokens[row] = out_tok
    return RerankBatch(ranked, input_tokens, output_tokens, latency)


class ChatCompletionReranker:
    def __init__(
        self,
        cache_dir: Path,
        model: str,
        provider: str,
        api_url: Optional[str],
        timeout: float,
        max_retries: int,
        max_output_tokens: int,
    ) -> None:
        env_name = "DEEPSEEK_API_KEY" if provider == "deepseek" else "OPENAI_API_KEY"
        api_key = os.environ.get(env_name)
        if not api_key:
            raise SystemExit(f"{env_name} is not set; use --mode proxy or export the key for --mode api.")
        default_url = (
            "https://api.deepseek.com/chat/completions"
            if provider == "deepseek"
            else "https://api.openai.com/v1/chat/completions"
        )
        self.cache_dir = cache_dir
        self.model = model
        self.provider = provider
        self.api_url = api_url or default_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_output_tokens = max_output_tokens
        self.api_key = api_key
        ensure_dirs(cache_dir)

    def _cache_path(self, prompt: str) -> Path:
        key = hashlib.sha256((self.provider + "\n" + self.model + "\n" + prompt).encode("utf-8")).hexdigest()
        return self.cache_dir / f"{key}.json"

    def _call(self, prompt: str) -> Dict[str, object]:
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a strict recommendation reranker. Return valid JSON only.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
            "max_tokens": self.max_output_tokens,
            "response_format": {"type": "json_object"},
        }
        if self.provider == "deepseek":
            payload["thinking"] = {"type": "disabled"}
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.api_url,
            data=data,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
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
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = exc
                sleep_seconds = min(2.0 ** attempt, 8.0)
                time.sleep(sleep_seconds)
        raise RuntimeError(f"Chat completion request failed after retries: {last_error}")

    def rerank_one(
        self,
        uid: int,
        history: np.ndarray,
        candidate_items: np.ndarray,
        meta: ItemMetadata,
        history_limit: int,
        topk: int,
        title_mask: bool,
    ) -> Tuple[np.ndarray, int, int, float, bool]:
        prompt = build_prompt(uid, history, candidate_items, meta, history_limit, title_mask)
        cache_path = self._cache_path(prompt)
        if cache_path.exists():
            record = json.loads(cache_path.read_text(encoding="utf-8"))
        else:
            record = self._call(prompt)
            cache_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

        content = str(record["choices"][0]["message"]["content"])
        usage = record.get("usage", {})
        input_tokens = int(usage.get("prompt_tokens", estimate_tokens(prompt)))
        output_tokens = int(usage.get("completion_tokens", estimated_output_tokens(topk)))
        latency = float(record.get("_latency_seconds", 0.0))
        code_to_item = {f"C{pos:02d}": int(iid) for pos, iid in enumerate(candidate_items, start=1)}
        parse_failure = False
        ranked_items: List[int] = []
        try:
            parsed = json.loads(content)
            raw_ranking = parsed.get("ranking", [])
            for code in raw_ranking:
                code_str = str(code).strip().upper()
                if code_str in code_to_item and code_to_item[code_str] not in ranked_items:
                    ranked_items.append(code_to_item[code_str])
        except json.JSONDecodeError:
            parse_failure = True

        if len(ranked_items) < topk:
            parse_failure = True
            for iid_np in candidate_items:
                iid = int(iid_np)
                if iid not in ranked_items:
                    ranked_items.append(iid)
                if len(ranked_items) >= topk:
                    break
        return np.asarray(ranked_items[:topk], dtype=np.int64), input_tokens, output_tokens, latency, parse_failure


def openai_rerank_batch(
    batch: CandidateBatch,
    meta: ItemMetadata,
    static_pop: np.ndarray,
    history_limit: int,
    topk: int,
    variant: str,
    seed: int,
    reranker: ChatCompletionReranker,
) -> RerankBatch:
    title_mask = variant == "title_mask"
    histories = build_perturbed_histories(batch.users, batch.histories, static_pop, variant, history_limit, seed)
    ranked = np.zeros((len(batch.users), topk), dtype=np.int64)
    input_tokens = np.zeros(len(batch.users), dtype=np.int32)
    output_tokens = np.zeros(len(batch.users), dtype=np.int32)
    latency = np.zeros(len(batch.users), dtype=np.float32)
    failures = 0
    for row, uid_np in enumerate(batch.users):
        uid = int(uid_np)
        ranked[row], input_tokens[row], output_tokens[row], latency[row], failed = reranker.rerank_one(
            uid, histories[uid], batch.candidates[row], meta, history_limit, topk, title_mask
        )
        failures += int(failed)
        if (row + 1) % 25 == 0 or row + 1 == len(batch.users):
            print(f"api reranked {row + 1}/{len(batch.users)} split={batch.split_name} variant={variant}", flush=True)
    return RerankBatch(ranked, input_tokens, output_tokens, latency, failures)


def compute_candidate_features(
    batch: CandidateBatch,
    meta: ItemMetadata,
    static_pop: np.ndarray,
    static_pct: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for row, uid_np in enumerate(batch.users):
        uid = int(uid_np)
        cand = batch.candidates[row]
        scores = batch.scores[row]
        hist = batch.histories[uid]
        target = int(batch.targets[row])
        target_positions = np.flatnonzero(cand == target)
        target_rank = int(target_positions[0]) + 1 if len(target_positions) else len(cand) + 1
        margin = float(scores[0] - scores[1]) if len(scores) > 1 and np.isfinite(scores[0] - scores[1]) else 0.0
        hist_pct = static_pct[hist] if len(hist) else np.asarray([], dtype=np.float32)
        rows.append(
            {
                "uid": uid,
                "target": target,
                "top1_score": float(scores[0]),
                "top1_top2_margin": margin,
                "score_entropy": softmax_entropy(scores),
                "history_length": int(len(hist)),
                "candidate_popularity_mean": float(np.mean(static_pop[cand])),
                "candidate_tail_ratio": float(np.mean(static_pct[cand] <= 0.20)),
                "candidate_metadata_length": float(np.mean(meta.text_lengths[cand])),
                "user_niche_score": float(1.0 - np.median(hist_pct)) if len(hist_pct) else 1.0,
                "baseline_target_rank": target_rank,
            }
        )
    return pd.DataFrame(rows)


def fit_standardizer(train_features: pd.DataFrame, columns: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    values = train_features[list(columns)].to_numpy(np.float32)
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    std[std < 1e-8] = 1.0
    return mean, std


def apply_standardizer(features: pd.DataFrame, columns: Sequence[str], mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (features[list(columns)].to_numpy(np.float32) - mean) / std


def rule_scores(val_features: pd.DataFrame, test_features: pd.DataFrame) -> np.ndarray:
    columns = [
        "score_entropy",
        "top1_top2_margin",
        "history_length",
        "candidate_tail_ratio",
        "candidate_metadata_length",
        "user_niche_score",
    ]
    mean, std = fit_standardizer(val_features, columns)
    x = apply_standardizer(test_features, columns, mean, std)
    entropy = x[:, 0]
    negative_margin = -x[:, 1]
    short_history = -x[:, 2]
    tail = x[:, 3]
    metadata = x[:, 4]
    niche = x[:, 5]
    return entropy + negative_margin + short_history + tail + metadata + niche


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -35.0, 35.0)))


def logistic_gate_scores(
    val_features: pd.DataFrame,
    val_labels: np.ndarray,
    test_features: pd.DataFrame,
    allow_leaky: bool,
    seed: int,
) -> np.ndarray:
    columns = [
        "top1_score",
        "top1_top2_margin",
        "score_entropy",
        "history_length",
        "candidate_popularity_mean",
        "candidate_tail_ratio",
        "candidate_metadata_length",
        "user_niche_score",
    ]
    if allow_leaky:
        columns.append("baseline_target_rank")
    y = val_labels.astype(np.float32)
    if len(np.unique(y)) < 2:
        return np.zeros(len(test_features), dtype=np.float32)
    mean, std = fit_standardizer(val_features, columns)
    x_val = apply_standardizer(val_features, columns, mean, std)
    x_test = apply_standardizer(test_features, columns, mean, std)
    x_val = np.column_stack([np.ones(len(x_val), dtype=np.float32), x_val])
    x_test = np.column_stack([np.ones(len(x_test), dtype=np.float32), x_test])
    rng = np.random.default_rng(seed)
    weights = rng.normal(0.0, 0.01, size=x_val.shape[1]).astype(np.float32)
    pos_weight = float(len(y) / max(1.0, 2.0 * float(y.sum())))
    neg_weight = float(len(y) / max(1.0, 2.0 * float((1.0 - y).sum())))
    sample_weight = np.where(y > 0.5, pos_weight, neg_weight).astype(np.float32)
    lr = 0.05
    l2 = 0.001
    denom = float(sample_weight.sum())
    for _ in range(1200):
        pred = sigmoid(x_val @ weights)
        grad = (x_val.T @ ((pred - y) * sample_weight)) / denom
        grad[1:] += l2 * weights[1:]
        weights -= lr * grad.astype(np.float32)
    return sigmoid(x_test @ weights).astype(np.float32)


def select_top_budget(scores: np.ndarray, budget: float) -> np.ndarray:
    n = len(scores)
    k = int(round(float(budget) * n))
    k = min(max(k, 0), n)
    mask = np.zeros(n, dtype=bool)
    if k <= 0:
        return mask
    order = np.argsort(-scores, kind="mergesort")
    mask[order[:k]] = True
    return mask


def metric_summary(metrics: pd.DataFrame) -> Dict[str, float]:
    return {
        "NDCG@20": float(metrics["NDCG@20"].mean()),
        "Recall@20": float(metrics["Recall@20"].mean()),
        "HitRate@20": float(metrics["HitRate@20"].mean()),
    }


def blended_metrics(base: pd.DataFrame, rerank: pd.DataFrame, invoke_mask: np.ndarray) -> pd.DataFrame:
    out = base.copy()
    for col in ["NDCG@20", "Recall@20", "HitRate@20"]:
        values = out[col].to_numpy(np.float64)
        rerank_values = rerank[col].to_numpy(np.float64)
        values[invoke_mask] = rerank_values[invoke_mask]
        out[col] = values
    return out


def token_cost(input_tokens: np.ndarray, output_tokens: np.ndarray, input_price: float, output_price: float) -> float:
    return float(input_tokens.sum()) * input_price / 1_000_000.0 + float(output_tokens.sum()) * output_price / 1_000_000.0


def cost_fields(
    invoke_mask: np.ndarray,
    input_tokens: np.ndarray,
    output_tokens: np.ndarray,
    latency: np.ndarray,
    input_price: float,
    output_price: float,
    always_cost: float,
    always_tokens: float,
) -> Dict[str, float]:
    selected_in = input_tokens[invoke_mask]
    selected_out = output_tokens[invoke_mask]
    selected_latency = latency[invoke_mask]
    total_tokens = float(selected_in.sum() + selected_out.sum())
    cost = token_cost(selected_in, selected_out, input_price, output_price)
    latency_per_user = float(selected_latency.sum() / len(invoke_mask)) if len(invoke_mask) else 0.0
    selected_cost_for_saving = cost if always_cost > 0 else total_tokens
    always_for_saving = always_cost if always_cost > 0 else always_tokens
    saving = 1.0 - selected_cost_for_saving / always_for_saving if always_for_saving > 0 else 0.0
    return {
        "Input Tokens/user": float(selected_in.sum() / len(invoke_mask)) if len(invoke_mask) else 0.0,
        "Output Tokens/user": float(selected_out.sum() / len(invoke_mask)) if len(invoke_mask) else 0.0,
        "Total Tokens": total_tokens,
        "Latency/user": latency_per_user,
        "Estimated Cost USD": cost,
        "Cost Saving": saving,
    }


def analyze_invocation_methods(
    baseline_name: str,
    val_features: pd.DataFrame,
    test_features: pd.DataFrame,
    val_base_metrics: pd.DataFrame,
    val_rerank_metrics: pd.DataFrame,
    test_base_metrics: pd.DataFrame,
    test_rerank_metrics: pd.DataFrame,
    rerank_tokens: RerankBatch,
    budgets: Sequence[float],
    random_trials: int,
    seed: int,
    input_price: float,
    output_price: float,
    allow_leaky: bool,
    invocation_name: str,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    n = len(test_base_metrics)
    all_mask = np.ones(n, dtype=bool)
    no_mask = np.zeros(n, dtype=bool)
    base_summary = metric_summary(test_base_metrics)
    always_summary = metric_summary(test_rerank_metrics)
    always_gain = always_summary["NDCG@20"] - base_summary["NDCG@20"]
    always_cost = token_cost(rerank_tokens.input_tokens, rerank_tokens.output_tokens, input_price, output_price)
    always_tokens = float(rerank_tokens.input_tokens.sum() + rerank_tokens.output_tokens.sum())

    rows: List[Dict[str, object]] = []
    always_method_name = f"Always {invocation_name}"
    masks: Dict[str, np.ndarray] = {"No LLM": no_mask, always_method_name: all_mask}

    def add_row(method: str, budget: float, invoke_mask: np.ndarray, metrics: pd.DataFrame) -> None:
        summary = metric_summary(metrics)
        gain = summary["NDCG@20"] - base_summary["NDCG@20"]
        capture = gain / always_gain if abs(always_gain) > 1e-12 else np.nan
        row: Dict[str, object] = {
            "Baseline": baseline_name,
            "Method": method,
            "Budget": float(budget),
            "Invocation Rate": float(invoke_mask.mean()) if len(invoke_mask) else 0.0,
            "NDCG@20": summary["NDCG@20"],
            "Recall@20": summary["Recall@20"],
            "HitRate@20": summary["HitRate@20"],
            "NDCG Gain": gain,
            "Benefit Capture": capture,
            "Random Gap": np.nan,
            "Oracle Gap": np.nan,
        }
        row.update(
            cost_fields(
                invoke_mask,
                rerank_tokens.input_tokens,
                rerank_tokens.output_tokens,
                rerank_tokens.latency_seconds,
                input_price,
                output_price,
                always_cost,
                always_tokens,
            )
        )
        rows.append(row)

    add_row("No LLM", 0.0, no_mask, test_base_metrics)
    add_row(always_method_name, 1.0, all_mask, test_rerank_metrics)

    val_labels = (val_rerank_metrics["NDCG@20"].to_numpy(np.float32) > val_base_metrics["NDCG@20"].to_numpy(np.float32)).astype(
        np.float32
    )
    learned_score = logistic_gate_scores(val_features, val_labels, test_features, allow_leaky, seed)
    rule_score = rule_scores(val_features, test_features)
    oracle_score = test_rerank_metrics["NDCG@20"].to_numpy(np.float32) - test_base_metrics["NDCG@20"].to_numpy(np.float32)

    random_ndcg_by_budget: Dict[float, float] = {}
    oracle_ndcg_by_budget: Dict[float, float] = {}

    for budget in budgets:
        if budget >= 1.0:
            random_ndcg_by_budget[float(budget)] = always_summary["NDCG@20"]
            oracle_ndcg_by_budget[float(budget)] = always_summary["NDCG@20"]
            continue
        rng = np.random.default_rng(seed + int(round(budget * 1000)))
        trial_metrics = []
        trial_input = []
        trial_output = []
        trial_latency = []
        for _ in range(random_trials):
            random_scores = rng.random(n)
            mask = select_top_budget(random_scores, budget)
            trial = blended_metrics(test_base_metrics, test_rerank_metrics, mask)
            trial_metrics.append(metric_summary(trial))
            trial_input.append(float(rerank_tokens.input_tokens[mask].sum()))
            trial_output.append(float(rerank_tokens.output_tokens[mask].sum()))
            trial_latency.append(float(rerank_tokens.latency_seconds[mask].sum() / n))
        random_ndcg = float(np.mean([m["NDCG@20"] for m in trial_metrics]))
        random_ndcg_by_budget[float(budget)] = random_ndcg
        random_mask = select_top_budget(rng.random(n), budget)
        random_metrics = blended_metrics(test_base_metrics, test_rerank_metrics, random_mask)
        add_row(f"Random@{int(round(budget * 100))}", budget, random_mask, random_metrics)
        rows[-1]["NDCG@20"] = random_ndcg
        rows[-1]["Recall@20"] = float(np.mean([m["Recall@20"] for m in trial_metrics]))
        rows[-1]["HitRate@20"] = float(np.mean([m["HitRate@20"] for m in trial_metrics]))
        rows[-1]["NDCG Gain"] = random_ndcg - base_summary["NDCG@20"]
        rows[-1]["Benefit Capture"] = (
            (random_ndcg - base_summary["NDCG@20"]) / always_gain if abs(always_gain) > 1e-12 else np.nan
        )
        rows[-1]["Input Tokens/user"] = float(np.mean(trial_input) / n)
        rows[-1]["Output Tokens/user"] = float(np.mean(trial_output) / n)
        rows[-1]["Total Tokens"] = float(np.mean(np.asarray(trial_input) + np.asarray(trial_output)))
        rows[-1]["Latency/user"] = float(np.mean(trial_latency))
        rows[-1]["Estimated Cost USD"] = (
            float(np.mean(trial_input)) * input_price / 1_000_000.0
            + float(np.mean(trial_output)) * output_price / 1_000_000.0
        )
        rows[-1]["Cost Saving"] = (
            1.0 - float(rows[-1]["Estimated Cost USD"]) / always_cost
            if always_cost > 0
            else 1.0 - float(rows[-1]["Total Tokens"]) / always_tokens
        )

        oracle_mask = select_top_budget(oracle_score, budget)
        oracle_metrics = blended_metrics(test_base_metrics, test_rerank_metrics, oracle_mask)
        oracle_ndcg_by_budget[float(budget)] = metric_summary(oracle_metrics)["NDCG@20"]

    for budget in budgets:
        rule_mask = select_top_budget(rule_score, budget)
        learned_mask = select_top_budget(learned_score, budget)
        oracle_mask = select_top_budget(oracle_score, budget)
        for method, mask in [
            (f"Rule-Selective@{int(round(budget * 100))}", rule_mask),
            (f"Learned-Selective@{int(round(budget * 100))}", learned_mask),
            (f"Oracle@{int(round(budget * 100))}", oracle_mask),
        ]:
            add_row(method, budget, mask, blended_metrics(test_base_metrics, test_rerank_metrics, mask))
            masks[method] = mask
            rows[-1]["Random Gap"] = float(rows[-1]["NDCG@20"] - random_ndcg_by_budget.get(float(budget), np.nan))
            rows[-1]["Oracle Gap"] = float(oracle_ndcg_by_budget.get(float(budget), np.nan) - rows[-1]["NDCG@20"])

    df = pd.DataFrame(rows)
    return df, masks


def subgroup_rows(
    baseline_name: str,
    features: pd.DataFrame,
    base_metrics: pd.DataFrame,
    rerank_metrics: pd.DataFrame,
    learned50_mask: np.ndarray,
    rerank_tokens: RerankBatch,
) -> pd.DataFrame:
    groups: Dict[str, np.ndarray] = {}
    q_hist_low = float(features["history_length"].quantile(0.25))
    q_hist_high = float(features["history_length"].quantile(0.75))
    q_niche_high = float(features["user_niche_score"].quantile(0.75))
    q_niche_low = float(features["user_niche_score"].quantile(0.25))
    q_tail_high = float(features["candidate_tail_ratio"].quantile(0.75))
    q_tail_low = float(features["candidate_tail_ratio"].quantile(0.25))
    q_entropy_high = float(features["score_entropy"].quantile(0.75))
    groups["short-history"] = features["history_length"].to_numpy() <= q_hist_low
    groups["long-history"] = features["history_length"].to_numpy() >= q_hist_high
    groups["niche"] = features["user_niche_score"].to_numpy() >= q_niche_high
    groups["mainstream"] = features["user_niche_score"].to_numpy() <= q_niche_low
    groups["cold-item candidate set"] = features["candidate_tail_ratio"].to_numpy() >= q_tail_high
    groups["warm-item candidate set"] = features["candidate_tail_ratio"].to_numpy() <= q_tail_low
    groups["low-confidence"] = features["score_entropy"].to_numpy() >= q_entropy_high

    rows = []
    for group, mask in groups.items():
        if not mask.any():
            continue
        always_gain = float(rerank_metrics.loc[mask, "NDCG@20"].mean() - base_metrics.loc[mask, "NDCG@20"].mean())
        blended = blended_metrics(base_metrics, rerank_metrics, learned50_mask)
        selective_gain = float(blended.loc[mask, "NDCG@20"].mean() - base_metrics.loc[mask, "NDCG@20"].mean())
        always_tokens = float((rerank_tokens.input_tokens[mask] + rerank_tokens.output_tokens[mask]).sum())
        selected = mask & learned50_mask
        selective_tokens = float((rerank_tokens.input_tokens[selected] + rerank_tokens.output_tokens[selected]).sum())
        rows.append(
            {
                "Baseline": baseline_name,
                "Group": group,
                "Users": int(mask.sum()),
                "Always LLM Gain": always_gain,
                "Selective Gain": selective_gain,
                "Invocation Rate": float(selected.sum() / mask.sum()),
                "Cost Saving": 1.0 - selective_tokens / always_tokens if always_tokens > 0 else 0.0,
            }
        )
    return pd.DataFrame(rows)


def diagnostic_rows(
    baseline_name: str,
    base_metrics: pd.DataFrame,
    variant_metrics: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    rows = []
    base_summary = metric_summary(base_metrics)
    for variant, metrics in variant_metrics.items():
        summary = metric_summary(metrics)
        rows.append(
            {
                "Baseline": baseline_name,
                "Diagnostic": variant,
                "NDCG@20": summary["NDCG@20"],
                "Recall@20": summary["Recall@20"],
                "HitRate@20": summary["HitRate@20"],
                "NDCG Gain vs No LLM": summary["NDCG@20"] - base_summary["NDCG@20"],
            }
        )
    return pd.DataFrame(rows)


def dataset_statistics(datadir: Path, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, n_users: int, n_items: int) -> pd.DataFrame:
    train_lengths = train.groupby("uid").size()
    return pd.DataFrame(
        [
            {
                "Dataset": datadir.name,
                "#Users": n_users,
                "#Items": n_items,
                "#Interactions": int(len(train) + len(val) + len(test)),
                "Metadata fields": "title|genres|year",
                "Avg history length": float(train_lengths.mean()),
                "Train": int(len(train)),
                "Validation": int(len(val)),
                "Test": int(len(test)),
            }
        ]
    )


def plot_cost_curve(cost_df: pd.DataFrame, baseline: str, figdir: Path) -> None:
    sub = cost_df[(cost_df["Baseline"] == baseline) & (cost_df["Budget"] > 0)].copy()
    if sub.empty:
        return
    families = [
        ("Random", "Random@"),
        ("Rule-Selective", "Rule-Selective@"),
        ("Learned-Selective", "Learned-Selective@"),
        ("Oracle", "Oracle@"),
    ]
    plt.figure(figsize=(7, 4.5))
    for label, prefix in families:
        part = sub[sub["Method"].str.startswith(prefix)].sort_values("Invocation Rate")
        if part.empty:
            continue
        plt.plot(part["Invocation Rate"], part["NDCG Gain"], marker="o", label=label)
    plt.axhline(0.0, color="#888888", linewidth=0.8)
    plt.xlabel("Invocation Rate")
    plt.ylabel("NDCG@20 Gain")
    plt.title(f"Cost-Utility Curve ({baseline})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figdir / f"figure1_cost_ndcg_curve_{baseline}.png", dpi=180)
    plt.close()


def plot_subgroup_heatmap(subgroup_df: pd.DataFrame, baseline: str, figdir: Path) -> None:
    sub = subgroup_df[subgroup_df["Baseline"] == baseline].copy()
    if sub.empty:
        return
    labels = sub["Group"].tolist()
    data = sub[["Always LLM Gain", "Selective Gain"]].to_numpy(np.float32)
    plt.figure(figsize=(7, max(3.5, 0.45 * len(labels))))
    plt.imshow(data, aspect="auto", cmap="RdYlGn")
    plt.yticks(np.arange(len(labels)), labels)
    plt.xticks([0, 1], ["Always", "Learned@50"])
    plt.colorbar(label="NDCG@20 gain")
    plt.title(f"Subgroup Utility ({baseline})")
    plt.tight_layout()
    plt.savefig(figdir / f"figure2_subgroup_utility_{baseline}.png", dpi=180)
    plt.close()


def plot_diagnostics(diag_df: pd.DataFrame, baseline: str, figdir: Path) -> None:
    sub = diag_df[diag_df["Baseline"] == baseline].copy()
    if sub.empty:
        return
    order = ["original", "title_mask", "shuffled_history", "random_history", "popular_history"]
    sub["order"] = sub["Diagnostic"].map({name: idx for idx, name in enumerate(order)})
    sub = sub.sort_values("order")
    plt.figure(figsize=(7, 4))
    plt.bar(sub["Diagnostic"], sub["NDCG Gain vs No LLM"], color="#4C78A8")
    plt.axhline(0.0, color="#888888", linewidth=0.8)
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("NDCG@20 Gain")
    plt.title(f"Masking and History Perturbation ({baseline})")
    plt.tight_layout()
    plt.savefig(figdir / f"figure3_diagnostics_{baseline}.png", dpi=180)
    plt.close()


def write_report(
    outdir: Path,
    mode: str,
    baselines: Sequence[str],
    dataset_df: pd.DataFrame,
    overall_df: pd.DataFrame,
    cost_df: pd.DataFrame,
    subgroup_df: pd.DataFrame,
    diag_df: pd.DataFrame,
    go_no_go: Mapping[str, object],
    allow_leaky: bool,
) -> None:
    def md_table(df: pd.DataFrame, max_rows: int = 12) -> str:
        if df.empty:
            return "(empty)"
        sample = df.head(max_rows).copy()
        columns = [str(col) for col in sample.columns]
        rendered_rows = []
        for row in sample.itertuples(index=False):
            rendered = []
            for value in row:
                if isinstance(value, float):
                    rendered.append(f"{value:.6g}")
                else:
                    rendered.append(str(value))
            rendered_rows.append(rendered)
        lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
        for rendered in rendered_rows:
            lines.append("| " + " | ".join(rendered) + " |")
        return "\n".join(lines)

    lines = [
        "# Selective LLM Invocation Pilot",
        "",
        f"Generated at: {datetime.now(timezone.utc).isoformat()}",
        f"Mode: `{mode}`",
        "",
    ]
    if mode == "proxy":
        lines.extend(
            [
                "**Important:** this run used the offline metadata-aware proxy reranker, not a real LLM.",
                "Use it to validate candidate generation, gating, cost accounting, and diagnostics before spending API budget.",
                "",
            ]
        )
    if allow_leaky:
        lines.extend(
            [
                "**Warning:** the learned gate included `baseline_target_rank`, which uses ground truth and is not deployable.",
                "",
            ]
        )
    lines.extend(["## Dataset", "", md_table(dataset_df), ""])
    lines.extend(["## Overall Performance", "", md_table(overall_df), ""])
    lines.extend(["## Cost-Utility Tradeoff", "", md_table(cost_df), ""])
    lines.extend(["## Subgroups", "", md_table(subgroup_df), ""])
    lines.extend(["## Diagnostics", "", md_table(diag_df), ""])
    lines.extend(["## Go / No-Go", "", "```json", json.dumps(go_no_go, indent=2), "```", ""])
    lines.extend(["## Figures", ""])
    for baseline in baselines:
        lines.append(f"- `figure1_cost_ndcg_curve_{baseline}.png`")
        lines.append(f"- `figure2_subgroup_utility_{baseline}.png`")
        lines.append(f"- `figure3_diagnostics_{baseline}.png`")
    (outdir / "pilot_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_go_no_go(
    mode: str,
    primary_baseline: str,
    cost_df: pd.DataFrame,
    subgroup_df: pd.DataFrame,
    diag_df: pd.DataFrame,
) -> Dict[str, object]:
    sub = cost_df[cost_df["Baseline"] == primary_baseline]
    always = sub[sub["Method"].isin(["Always LLM", "Always Proxy"])]
    learned50 = sub[sub["Method"] == "Learned-Selective@50"]
    random50 = sub[sub["Method"] == "Random@50"]
    criteria: Dict[str, object] = {"primary_baseline": primary_baseline, "mode": mode}
    if always.empty or learned50.empty or random50.empty:
        criteria["decision"] = "incomplete"
        return criteria
    always_gain = float(always["NDCG Gain"].iloc[0])
    learned_capture = float(learned50["Benefit Capture"].iloc[0])
    learned_random_gap = float(learned50["NDCG@20"].iloc[0] - random50["NDCG@20"].iloc[0])
    cost_saving = float(learned50["Cost Saving"].iloc[0])
    group_sub = subgroup_df[subgroup_df["Baseline"] == primary_baseline]
    max_group_gain = float(group_sub["Always LLM Gain"].max()) if not group_sub.empty else 0.0
    diag_sub = diag_df[diag_df["Baseline"] == primary_baseline]
    original_gain = float(
        diag_sub.loc[diag_sub["Diagnostic"] == "original", "NDCG Gain vs No LLM"].iloc[0]
    ) if not diag_sub[diag_sub["Diagnostic"] == "original"].empty else 0.0
    min_diag_gain = float(diag_sub["NDCG Gain vs No LLM"].min()) if not diag_sub.empty else original_gain
    criteria.update(
        {
            "always_gain_positive": always_gain > 0.0,
            "always_gain": always_gain,
            "learned50_capture_ge_70pct": learned_capture >= 0.70 if not math.isnan(learned_capture) else False,
            "learned50_benefit_capture": learned_capture,
            "learned50_beats_random50": learned_random_gap > 0.0,
            "learned50_random_gap": learned_random_gap,
            "subgroup_gain_exceeds_overall": max_group_gain > always_gain,
            "max_subgroup_always_gain": max_group_gain,
            "cost_saving_ge_40pct": cost_saving >= 0.40,
            "learned50_cost_saving": cost_saving,
            "diagnostic_changes_gain": abs(original_gain - min_diag_gain) > 1e-12,
            "min_diagnostic_gain": min_diag_gain,
        }
    )
    pass_count = sum(
        bool(criteria[key])
        for key in [
            "always_gain_positive",
            "learned50_capture_ge_70pct",
            "learned50_beats_random50",
            "subgroup_gain_exceeds_overall",
            "cost_saving_ge_40pct",
            "diagnostic_changes_gain",
        ]
    )
    if mode == "proxy":
        criteria["decision"] = "proxy_only_not_valid_for_true_llm_go_no_go"
        criteria["proxy_pass_count"] = pass_count
    else:
        criteria["decision"] = "go" if pass_count == 6 else "no_go_or_revise"
    return criteria


def generate_batches_for_baseline(
    baseline: str,
    val_eval: pd.DataFrame,
    test_eval: pd.DataFrame,
    train: pd.DataFrame,
    val: pd.DataFrame,
    train_histories: Sequence[np.ndarray],
    test_histories: Sequence[np.ndarray],
    val_exclude: Sequence[np.ndarray],
    test_exclude: Sequence[np.ndarray],
    static_pop: np.ndarray,
    n_users: int,
    n_items: int,
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[CandidateBatch, CandidateBatch]:
    topn = args.top_candidates
    if baseline == "pop":
        val_batch = generate_pop_candidates(val_eval, train_histories, val_exclude, static_pop, topn, "val")
        test_batch = generate_pop_candidates(test_eval, test_histories, test_exclude, static_pop, topn, "test")
        return val_batch, test_batch
    if baseline == "lightgcn":
        user_emb, item_emb = load_lightgcn_embeddings(args.config, args.lightgcn_ckpt, train, n_users, n_items, device)
        val_batch = generate_lightgcn_candidates(
            val_eval, train_histories, val_exclude, user_emb, item_emb, topn, args.eval_batch_size, "val"
        )
        test_batch = generate_lightgcn_candidates(
            test_eval, test_histories, test_exclude, user_emb, item_emb, topn, args.eval_batch_size, "test"
        )
        return val_batch, test_batch
    if baseline == "sasrec":
        model = load_sasrec_model(args.sasrec_ckpt, n_items, device)
        train_sequences = build_ordered_user_sequences(train, n_users)
        test_sequences = build_ordered_user_sequences(pd.concat([train, val], ignore_index=True), n_users)
        val_batch = generate_sasrec_candidates(
            val_eval,
            train_histories,
            val_exclude,
            train_sequences,
            model,
            n_items,
            topn,
            args.eval_batch_size,
            device,
            "val",
        )
        test_batch = generate_sasrec_candidates(
            test_eval,
            test_histories,
            test_exclude,
            test_sequences,
            model,
            n_items,
            topn,
            args.eval_batch_size,
            device,
            "test",
        )
        return val_batch, test_batch
    raise ValueError(f"Unknown baseline: {baseline}")


def rerank_variants(
    batch: CandidateBatch,
    meta: ItemMetadata,
    static_pop: np.ndarray,
    args: argparse.Namespace,
    reranker: Optional[ChatCompletionReranker],
    split_role: str,
) -> Dict[str, RerankBatch]:
    if args.mode == "api" and split_role == "diagnostic" and not args.force_api_diagnostics:
        variants = ["original"]
    elif split_role == "val":
        variants = ["original"]
    else:
        variants = ["original", "title_mask", "shuffled_history", "random_history", "popular_history"]

    out: Dict[str, RerankBatch] = {}
    for variant in variants:
        if args.mode == "proxy":
            out[variant] = proxy_rerank_batch(
                batch,
                meta,
                static_pop,
                args.history_limit,
                args.topk,
                variant,
                args.seed,
                args.proxy_latency_seconds,
            )
        else:
            if reranker is None:
                raise RuntimeError("API reranker is not initialized")
            out[variant] = openai_rerank_batch(
                batch,
                meta,
                static_pop,
                args.history_limit,
                args.topk,
                variant,
                args.seed,
                reranker,
            )
    return out


def save_candidates(outdir: Path, baseline: str, batch: CandidateBatch) -> None:
    np.savez_compressed(
        outdir / f"candidates_{baseline}_{batch.split_name}.npz",
        users=batch.users,
        targets=batch.targets,
        candidates=batch.candidates,
        scores=batch.scores,
    )


def main() -> None:
    args = parse_args()
    ensure_dirs(args.outdir, args.figdir)
    set_seed(args.seed)
    train, val, test, _ = read_interaction_split(args.datadir)
    n_users, n_items = infer_shape(train, val, test)
    static_pop = static_popularity(train, n_items)
    static_bucket = assign_buckets(static_pop, static_pop, dormant_for_zero=False)
    static_pct = popularity_percentiles(static_pop, static_pop)
    _ = static_bucket

    train_histories = build_user_histories(train, n_users)
    test_history_frame = pd.concat([train, val], ignore_index=True)
    test_histories = build_ordered_user_sequences(test_history_frame, n_users)
    val_exclude = build_exclude_lists(train_histories, None, n_users)
    test_exclude = build_exclude_lists(train_histories, val, n_users)
    groups = activity_controlled_user_groups(train_histories, static_pct)
    _ = groups
    meta = read_item_metadata(args.movies_path, args.datadir / "mappings.json", n_items)
    val_eval = sample_eval_frame(val, args.max_users, args.seed + 1)
    test_eval = sample_eval_frame(test, args.max_users, args.seed + 2)
    device = device_from_arg(args.device)

    cache_dir = args.cache_dir or (args.outdir / "api_cache")
    reranker = None
    if args.mode == "api":
        reranker = ChatCompletionReranker(
            cache_dir,
            args.model,
            args.provider,
            args.api_url,
            args.openai_timeout,
            args.openai_max_retries,
            args.openai_max_output_tokens,
        )

    dataset_df = dataset_statistics(args.datadir, train, val, test, n_users, n_items)
    dataset_df.to_csv(args.outdir / "table1_dataset_statistics.csv", index=False)

    overall_rows: List[Dict[str, object]] = []
    all_cost_rows: List[pd.DataFrame] = []
    all_subgroup_rows: List[pd.DataFrame] = []
    all_diag_rows: List[pd.DataFrame] = []
    feature_frames: List[pd.DataFrame] = []

    for baseline in args.baselines:
        print(f"=== Baseline: {baseline} ===", flush=True)
        val_batch, test_batch = generate_batches_for_baseline(
            baseline,
            val_eval,
            test_eval,
            train,
            val,
            train_histories,
            test_histories,
            val_exclude,
            test_exclude,
            static_pop,
            n_users,
            n_items,
            args,
            device,
        )
        save_candidates(args.outdir, baseline, val_batch)
        save_candidates(args.outdir, baseline, test_batch)

        val_base_metrics = baseline_metrics_from_ranked(val_batch.candidates[:, : args.topk], val_batch.targets)
        test_base_metrics = baseline_metrics_from_ranked(test_batch.candidates[:, : args.topk], test_batch.targets)
        val_features = compute_candidate_features(val_batch, meta, static_pop, static_pct)
        test_features = compute_candidate_features(test_batch, meta, static_pop, static_pct)
        val_features.insert(0, "Baseline", baseline)
        val_features.insert(1, "Split", "val")
        test_features.insert(0, "Baseline", baseline)
        test_features.insert(1, "Split", "test")
        feature_frames.extend([val_features, test_features])
        val_features_model = val_features.drop(columns=["Baseline", "Split"])
        test_features_model = test_features.drop(columns=["Baseline", "Split"])

        val_reranks = rerank_variants(val_batch, meta, static_pop, args, reranker, "val")
        test_reranks = rerank_variants(test_batch, meta, static_pop, args, reranker, "diagnostic")
        val_rerank_metrics = baseline_metrics_from_ranked(val_reranks["original"].ranked_topk, val_batch.targets)
        test_variant_metrics = {
            variant: baseline_metrics_from_ranked(batch.ranked_topk, test_batch.targets)
            for variant, batch in test_reranks.items()
        }
        test_rerank_metrics = test_variant_metrics["original"]

        for method_name, metrics, invoke_rate, tokens in [
            ("No LLM", test_base_metrics, 0.0, 0.0),
            (
                "Always LLM" if args.mode == "api" else "Always Proxy",
                test_rerank_metrics,
                1.0,
                float(test_reranks["original"].input_tokens.sum() + test_reranks["original"].output_tokens.sum()),
            ),
        ]:
            summary = metric_summary(metrics)
            overall_rows.append(
                {
                    "Baseline": baseline,
                    "Method": method_name,
                    "NDCG@20": summary["NDCG@20"],
                    "Recall@20": summary["Recall@20"],
                    "HitRate@20": summary["HitRate@20"],
                    "Invocation Rate": invoke_rate,
                    "Tokens": tokens,
                    "Latency": float(test_reranks["original"].latency_seconds.mean()) if invoke_rate else 0.0,
                }
            )

        cost_df, masks = analyze_invocation_methods(
            baseline,
            val_features_model,
            test_features_model,
            val_base_metrics,
            val_rerank_metrics,
            test_base_metrics,
            test_rerank_metrics,
            test_reranks["original"],
            args.budgets,
            args.random_trials,
            args.seed,
            args.input_price_per_1m,
            args.output_price_per_1m,
            args.allow_leaky_gate_feature,
            "LLM" if args.mode == "api" else "Proxy",
        )
        all_cost_rows.append(cost_df)

        learned50_mask = masks.get("Learned-Selective@50")
        if learned50_mask is None:
            learned50_mask = select_top_budget(np.zeros(len(test_base_metrics), dtype=np.float32), 0.50)
        all_subgroup_rows.append(
            subgroup_rows(baseline, test_features_model, test_base_metrics, test_rerank_metrics, learned50_mask, test_reranks["original"])
        )
        all_diag_rows.append(diagnostic_rows(baseline, test_base_metrics, test_variant_metrics))

        for variant, rerank_batch in test_reranks.items():
            np.save(args.outdir / f"reranked_{baseline}_{variant}_top{args.topk}.npy", rerank_batch.ranked_topk)
            pd.DataFrame(
                {
                    "uid": test_batch.users,
                    "input_tokens": rerank_batch.input_tokens,
                    "output_tokens": rerank_batch.output_tokens,
                    "latency_seconds": rerank_batch.latency_seconds,
                }
            ).to_csv(args.outdir / f"cost_trace_{baseline}_{variant}.csv", index=False)

    overall_df = pd.DataFrame(overall_rows)
    cost_utility_df = pd.concat(all_cost_rows, ignore_index=True)
    subgroup_df = pd.concat(all_subgroup_rows, ignore_index=True)
    diag_df = pd.concat(all_diag_rows, ignore_index=True)
    feature_df = pd.concat(feature_frames, ignore_index=True)

    overall_df.to_csv(args.outdir / "table2_overall_performance.csv", index=False)
    cost_utility_df.to_csv(args.outdir / "table3_cost_utility_tradeoff.csv", index=False)
    subgroup_df.to_csv(args.outdir / "table4_subgroup_analysis.csv", index=False)
    diag_df.to_csv(args.outdir / "table5_diagnostics.csv", index=False)
    feature_df.to_csv(args.outdir / "candidate_features.csv", index=False)

    primary = "lightgcn" if "lightgcn" in args.baselines else args.baselines[0]
    for baseline in args.baselines:
        plot_cost_curve(cost_utility_df, baseline, args.figdir)
        plot_subgroup_heatmap(subgroup_df, baseline, args.figdir)
        plot_diagnostics(diag_df, baseline, args.figdir)

    go_no_go = build_go_no_go(args.mode, primary, cost_utility_df, subgroup_df, diag_df)
    run_manifest = {
        "status": "completed",
        "mode": args.mode,
        "provider": args.provider if args.mode == "api" else None,
        "model": args.model if args.mode == "api" else None,
        "seed": args.seed,
        "datadir": str(args.datadir),
        "movies_path": str(args.movies_path),
        "baselines": args.baselines,
        "top_candidates": args.top_candidates,
        "topk": args.topk,
        "history_limit": args.history_limit,
        "budgets": args.budgets,
        "max_users": args.max_users,
        "input_price_per_1m": args.input_price_per_1m,
        "output_price_per_1m": args.output_price_per_1m,
        "allow_leaky_gate_feature": args.allow_leaky_gate_feature,
        "go_no_go": go_no_go,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (args.outdir / "go_no_go.json").write_text(json.dumps(go_no_go, indent=2) + "\n", encoding="utf-8")
    (args.outdir / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2) + "\n", encoding="utf-8")
    write_report(
        args.outdir,
        args.mode,
        args.baselines,
        dataset_df,
        overall_df,
        cost_utility_df,
        subgroup_df,
        diag_df,
        go_no_go,
        args.allow_leaky_gate_feature,
    )
    print(f"Done. Report: {args.outdir / 'pilot_report.md'}", flush=True)


if __name__ == "__main__":
    main()

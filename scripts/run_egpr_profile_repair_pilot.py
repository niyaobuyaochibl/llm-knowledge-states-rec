#!/usr/bin/env python3
"""ML-1M Evidence-Grounded Profile Repair mini-pilot.

This script implements the first closed loop for the EGPR direction:

1. Build recent user histories and LightGCN top-N candidate sets.
2. Generate raw profile claims, either through a chat-completion API or a
   deterministic metadata proxy for plumbing checks.
3. Score each claim against interaction evidence from the user's training
   history.
4. Evaluate Raw Profile, Remove Repair, and Evidence-Weighted Repair as
   profile-assisted rerankers over LightGCN candidates.

The metadata proxy is explicitly not an LLM result. It exists to validate the
pipeline when API credentials are unavailable.
"""

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
import torch

ROOT = Path("/root/temporal_popularity_pilot")
SRC = ROOT / "src"
SCRIPT_DIR = ROOT / "scripts"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SCRIPT_DIR))

from temporal_popularity.data import (  # noqa: E402
    build_exclude_lists,
    build_user_histories,
    infer_shape,
    read_interaction_split,
    set_seed,
)
from temporal_popularity.eval import topk_indices  # noqa: E402
from temporal_popularity.model import LightGCN, build_norm_adj  # noqa: E402

from run_llm_selective_invocation_pilot import (  # noqa: E402
    ItemMetadata,
    device_from_arg,
    ensure_dirs,
    estimate_tokens,
    read_item_metadata,
    sample_eval_frame,
    zscore,
)


GENRE_ALIASES: Dict[str, Tuple[str, ...]] = {
    "action": ("action", "action-packed", "explosive"),
    "adventure": ("adventure", "adventurous", "quest", "journey"),
    "animation": ("animation", "animated", "cartoon"),
    "children's": ("children", "childrens", "kids", "kid", "family", "families"),
    "comedy": ("comedy", "comedies", "comedic", "funny", "humorous", "humor", "lighthearted", "light-hearted"),
    "crime": ("crime", "criminal", "gangster", "detective"),
    "documentary": ("documentary", "documentaries", "nonfiction", "non-fiction"),
    "drama": ("drama", "dramatic", "serious", "character-driven"),
    "fantasy": ("fantasy", "magical", "magic", "mythic", "mythical"),
    "film-noir": ("noir", "film-noir"),
    "horror": ("horror", "scary", "frightening", "supernatural"),
    "musical": ("musical", "music", "song", "songs"),
    "mystery": ("mystery", "mysterious", "whodunit"),
    "romance": ("romance", "romantic", "love"),
    "sci-fi": ("sci-fi", "scifi", "science fiction", "science-fiction", "futuristic", "space"),
    "thriller": ("thriller", "thrillers", "suspense", "suspenseful"),
    "war": ("war", "military", "battle"),
    "western": ("western", "cowboy"),
}

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "based",
    "by",
    "enjoys",
    "films",
    "for",
    "from",
    "in",
    "likes",
    "movies",
    "of",
    "on",
    "prefers",
    "stories",
    "that",
    "the",
    "to",
    "with",
}

PROXY_UNSUPPORTED_CLAIMS: Tuple[Tuple[str, str], ...] = (
    ("likes science fiction thrillers", "theme"),
    ("prefers romantic period dramas", "genre"),
    ("enjoys horror and supernatural suspense", "theme"),
    ("likes musical performance stories", "style"),
    ("prefers western frontier adventures", "genre"),
)


@dataclass
class CandidateBatch:
    users: np.ndarray
    targets: np.ndarray
    candidates: np.ndarray
    scores: np.ndarray
    split_name: str


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
    parser.add_argument("--outdir", type=Path, default=ROOT / "results/egpr_profile_repair/ml1m_seed42_proxy")
    parser.add_argument("--profile-mode", choices=["metadata_proxy", "api"], default="metadata_proxy")
    parser.add_argument("--provider", choices=["deepseek", "openai"], default="deepseek")
    parser.add_argument("--model", default=None)
    parser.add_argument("--api-url", default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--api-timeout", type=float, default=60.0)
    parser.add_argument("--api-max-retries", type=int, default=3)
    parser.add_argument("--api-max-output-tokens", type=int, default=500)
    parser.add_argument("--max-users", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--top-candidates", type=int, default=100)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--history-limit", type=int, default=20)
    parser.add_argument("--claims-per-user", type=int, default=5)
    parser.add_argument(
        "--prompt-variant",
        choices=["conservative", "expressive", "overgeneralizing"],
        default="conservative",
        help="Profile-generation prompt style for stress testing.",
    )
    parser.add_argument("--proxy-noise-claims", type=int, default=1)
    parser.add_argument("--support-threshold", type=float, default=0.25)
    parser.add_argument(
        "--lambda-grid",
        nargs="+",
        type=float,
        default=[0.05, 0.1, 0.2, 0.3, 0.5, 1.0],
    )
    args = parser.parse_args()
    if args.model is None:
        args.model = "deepseek-v4-flash" if args.provider == "deepseek" else "gpt-4.1-mini"
    return args


def normalize_genre(genre: str) -> str:
    return genre.strip().lower()


def simple_tokens(text: str) -> Tuple[str, ...]:
    tokens: List[str] = []
    lowered = text.lower().replace("'", "")
    lowered = lowered.replace("sci fi", "sci-fi").replace("science-fiction", "science fiction")
    for token in re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)?", lowered):
        if len(token) > 1 and token not in STOPWORDS and not token.isdigit():
            tokens.append(token)
    return tuple(sorted(set(tokens)))


def genre_words_for_claim(claim: str) -> Tuple[str, ...]:
    lowered = claim.lower().replace("'", "")
    matched: List[str] = []
    for genre, aliases in GENRE_ALIASES.items():
        for alias in aliases:
            if alias in lowered:
                matched.append(genre)
                break
    return tuple(sorted(set(matched)))


def temporal_support_for_claim(claim: str, iid: int, meta: ItemMetadata) -> Optional[float]:
    if np.isnan(meta.years[iid]):
        return 0.0
    year = int(meta.years[iid])
    lowered = claim.lower()
    scores: List[float] = []
    for match in re.finditer(r"(?:\b(early|mid|late)\s+)?(\d{4})s", lowered):
        qualifier = match.group(1)
        decade = int(match.group(2))
        if qualifier == "early":
            start, end = decade, decade + 4
        elif qualifier == "mid":
            start, end = decade + 3, decade + 6
        elif qualifier == "late":
            start, end = decade + 5, decade + 9
        else:
            start, end = decade, decade + 9
        scores.append(1.0 if start <= year <= end else 0.0)
    for match in re.finditer(r"(?:year|from|in)\s+(\d{4})(?!s)", lowered):
        target = int(match.group(1))
        scores.append(1.0 if year == target else 0.0)
    for match in re.finditer(r"pre[- ]?(\d{4})", lowered):
        target = int(match.group(1))
        scores.append(1.0 if year < target else 0.0)
    for match in re.finditer(r"post[- ]?(\d{4})", lowered):
        target = int(match.group(1))
        scores.append(1.0 if year > target else 0.0)
    if "older" in lowered or "classic" in lowered:
        scores.append(1.0 if year <= 1980 else (0.5 if year <= 1990 else 0.0))
    if "contemporary" in lowered or "modern" in lowered:
        scores.append(1.0 if year >= 1990 else 0.0)
    if not scores:
        return None
    return max(scores)


def structural_support_for_claim(claim: str, iid: int, meta: ItemMetadata) -> Optional[float]:
    lowered = claim.lower()
    markers = ("multiple genres", "mix of genres", "multi-genre", "hybrid", "hybrids", "combined with")
    if not any(marker in lowered for marker in markers):
        return None
    return 1.0 if len(meta.genres[iid]) >= 2 else 0.0


def item_text_tokens(iid: int, meta: ItemMetadata) -> Tuple[str, ...]:
    values: List[str] = list(meta.title_tokens[iid])
    for genre in meta.genres[iid]:
        values.extend(simple_tokens(genre))
    return tuple(sorted(set(values)))


def genre_label(genre: str) -> str:
    labels = {
        "sci-fi": "science fiction",
        "children's": "family",
        "film-noir": "film noir",
    }
    return labels.get(genre, genre)


def cosine_sparse(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    if not left or not right:
        return 0.0
    common = set(left).intersection(right)
    numerator = sum(float(left[k]) * float(right[k]) for k in common)
    left_norm = math.sqrt(sum(float(v) * float(v) for v in left.values()))
    right_norm = math.sqrt(sum(float(v) * float(v) for v in right.values()))
    if left_norm <= 0.0 or right_norm <= 0.0:
        return 0.0
    return numerator / (left_norm * right_norm)


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def build_ordered_histories(frame: pd.DataFrame, n_users: int) -> List[np.ndarray]:
    rows = frame.sort_values(["uid", "timestamp", "iid"], kind="mergesort")
    histories: List[List[int]] = [[] for _ in range(n_users)]
    for uid, iid in rows[["uid", "iid"]].itertuples(index=False):
        histories[int(uid)].append(int(iid))
    return [np.asarray(items, dtype=np.int64) for items in histories]


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


def generate_lightgcn_candidates(
    eval_frame: pd.DataFrame,
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
            scores_row = score_batch[local].astype(np.float32).copy()
            excluded = exclude_lists[uid]
            if len(excluded):
                scores_row[excluded] = -np.inf
            idx = topk_indices(scores_row, topn)
            chosen_scores = scores_row[idx].astype(np.float32)
            finite = np.isfinite(chosen_scores)
            if not finite.all():
                fill = float(chosen_scores[finite].min() - 1.0) if finite.any() else 0.0
                chosen_scores = np.where(finite, chosen_scores, fill).astype(np.float32)
            candidates[start + local] = idx.astype(np.int64)
            scores[start + local] = chosen_scores
    return CandidateBatch(users=users, targets=targets, candidates=candidates, scores=scores, split_name=split_name)


def save_candidates(outdir: Path, batch: CandidateBatch) -> None:
    np.savez_compressed(
        outdir / f"candidates_lightgcn_{batch.split_name}.npz",
        users=batch.users,
        targets=batch.targets,
        candidates=batch.candidates,
        scores=batch.scores,
    )


def history_lines(uid: int, ordered_histories: Sequence[np.ndarray], meta: ItemMetadata, history_limit: int) -> List[str]:
    recent = ordered_histories[uid][-history_limit:]
    lines = []
    for pos, iid_np in enumerate(recent, start=1):
        iid = int(iid_np)
        genres = "|".join(meta.genres[iid]) if meta.genres[iid] else "Unknown"
        lines.append(f"{pos}. {meta.titles[iid]} | {genres}")
    return lines


def write_user_histories(
    outdir: Path,
    users: Iterable[int],
    ordered_histories: Sequence[np.ndarray],
    meta: ItemMetadata,
    history_limit: int,
) -> None:
    path = outdir / "user_history.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for uid in sorted(set(int(u) for u in users)):
            items = []
            for iid_np in ordered_histories[uid][-history_limit:]:
                iid = int(iid_np)
                items.append({"iid": iid, "title": meta.titles[iid], "genres": list(meta.genres[iid])})
            handle.write(json.dumps({"uid": uid, "history": items}, ensure_ascii=False) + "\n")


def build_profile_prompt(uid: int, ordered_histories: Sequence[np.ndarray], meta: ItemMetadata, history_limit: int, claims: int, prompt_variant: str) -> str:
    if prompt_variant == "conservative":
        lines = [
            f"Given the following MovieLens user interaction history, generate exactly {claims} concise preference claims.",
            "Rules:",
            "1. Each claim should be short and specific.",
            "2. Do not mention item names.",
            "3. Do not infer preferences that are not supported by the history.",
            "4. Return valid JSON only with schema {\"claims\":[{\"claim\":\"...\",\"type\":\"genre/style/theme/preference\",\"confidence\":0.0}]}",
            "",
            f"User: U{uid}",
            "User history:",
        ]
    elif prompt_variant == "expressive":
        lines = [
            f"Given the following MovieLens user interaction history, generate exactly {claims} rich user preference claims.",
            "Include explicit and latent preferences that plausibly explain the user's history.",
            "Cover genre, style, theme, mood, and possible viewing intentions when useful.",
            "Rules:",
            "1. Each claim should be concise but more expressive than a genre label.",
            "2. Do not mention item names.",
            "3. You may infer broader latent preferences from recurring patterns.",
            "4. Return valid JSON only with schema {\"claims\":[{\"claim\":\"...\",\"type\":\"genre/style/theme/mood/intent/preference\",\"confidence\":0.0}]}",
            "",
            f"User: U{uid}",
            "User history:",
        ]
    elif prompt_variant == "overgeneralizing":
        lines = [
            f"Given the following MovieLens user interaction history, infer broader latent interests and generate exactly {claims} preference claims.",
            "Go beyond direct genre repetition. Propose high-level interests, moods, themes, franchises, eras, and viewing intentions that may explain the history.",
            "Rules:",
            "1. Make the profile broad and hypothesis-rich rather than conservative.",
            "2. Do not mention item names.",
            "3. Include plausible latent interests even when they are not directly stated by metadata.",
            "4. Return valid JSON only with schema {\"claims\":[{\"claim\":\"...\",\"type\":\"genre/style/theme/mood/intent/preference\",\"confidence\":0.0}]}",
            "",
            f"User: U{uid}",
            "User history:",
        ]
    else:
        raise ValueError(f"Unknown prompt variant: {prompt_variant}")
    lines.extend(history_lines(uid, ordered_histories, meta, history_limit) or ["None"])
    return "\n".join(lines)


class ChatProfileGenerator:
    def __init__(
        self,
        cache_dir: Path,
        provider: str,
        model: str,
        api_url: Optional[str],
        timeout: float,
        max_retries: int,
        max_output_tokens: int,
        prompt_variant: str,
    ) -> None:
        env_name = "DEEPSEEK_API_KEY" if provider == "deepseek" else "OPENAI_API_KEY"
        api_key = os.environ.get(env_name)
        if not api_key:
            raise SystemExit(f"{env_name} is not set; use --profile-mode metadata_proxy or export the key.")
        default_url = (
            "https://api.deepseek.com/chat/completions"
            if provider == "deepseek"
            else "https://api.openai.com/v1/chat/completions"
        )
        self.cache_dir = cache_dir
        self.provider = provider
        self.model = model
        self.api_url = api_url or default_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_output_tokens = max_output_tokens
        self.prompt_variant = prompt_variant
        self.api_key = api_key
        ensure_dirs(cache_dir)

    def _cache_path(self, prompt: str) -> Path:
        key = hashlib.sha256((self.provider + "\n" + self.model + "\n" + prompt).encode("utf-8")).hexdigest()
        return self.cache_dir / f"{key}.json"

    def _call(self, prompt: str) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You generate strict JSON user preference claims for recommendation research.",
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
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, http.client.RemoteDisconnected, ConnectionResetError) as exc:
                last_error = exc
                time.sleep(min(2.0**attempt, 8.0))
        raise RuntimeError(f"Profile request failed after retries: {last_error}")

    def generate_one(
        self,
        uid: int,
        ordered_histories: Sequence[np.ndarray],
        meta: ItemMetadata,
        history_limit: int,
        claims_per_user: int,
    ) -> Tuple[List[Dict[str, object]], Dict[str, float]]:
        prompt = build_profile_prompt(uid, ordered_histories, meta, history_limit, claims_per_user, self.prompt_variant)
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
        for entry in raw_claims:
            if not isinstance(entry, dict):
                continue
            claim = str(entry.get("claim", "")).strip()
            if not claim:
                continue
            claims.append(
                {
                    "claim": claim,
                    "type": str(entry.get("type", "preference")).strip() or "preference",
                    "confidence": float(entry.get("confidence", 0.0) or 0.0),
                }
            )
            if len(claims) >= claims_per_user:
                break
        usage = record.get("usage", {})
        cost = {
            "input_tokens": float(usage.get("prompt_tokens", estimate_tokens(prompt))),
            "output_tokens": float(usage.get("completion_tokens", estimate_tokens(content))),
            "latency_seconds": float(record.get("_latency_seconds", 0.0)),
        }
        return claims, cost


def metadata_proxy_claims(
    uid: int,
    ordered_histories: Sequence[np.ndarray],
    meta: ItemMetadata,
    history_limit: int,
    claims_per_user: int,
    noise_claims: int,
) -> List[Dict[str, object]]:
    recent = ordered_histories[uid][-history_limit:]
    genre_counts: Dict[str, int] = {}
    co_counts: Dict[Tuple[str, str], int] = {}
    years: List[float] = []
    for iid_np in recent:
        iid = int(iid_np)
        genres = [normalize_genre(g) for g in meta.genres[iid]]
        for genre in genres:
            genre_counts[genre] = genre_counts.get(genre, 0) + 1
        for left_idx, left in enumerate(genres):
            for right in genres[left_idx + 1 :]:
                pair = tuple(sorted((left, right)))
                co_counts[pair] = co_counts.get(pair, 0) + 1
        if not np.isnan(meta.years[iid]):
            years.append(float(meta.years[iid]))

    ordered_genres = sorted(genre_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    claims: List[Dict[str, object]] = []
    if "animation" in genre_counts and "children's" in genre_counts:
        claims.append({"claim": "likes animated family movies", "type": "genre", "confidence": 0.0})
    if "action" in genre_counts and "adventure" in genre_counts:
        claims.append({"claim": "enjoys action adventure films", "type": "theme", "confidence": 0.0})
    if "comedy" in genre_counts:
        claims.append({"claim": "prefers humorous light-hearted stories", "type": "style", "confidence": 0.0})
    if "drama" in genre_counts:
        claims.append({"claim": "likes dramatic character-driven films", "type": "style", "confidence": 0.0})

    for genre, _ in ordered_genres:
        text = f"likes {genre_label(genre)} movies"
        if not any(existing["claim"] == text for existing in claims):
            claims.append({"claim": text, "type": "genre", "confidence": 0.0})
        if len(claims) >= max(0, claims_per_user - noise_claims):
            break

    if len(claims) < max(0, claims_per_user - noise_claims):
        for pair, _ in sorted(co_counts.items(), key=lambda kv: (-kv[1], kv[0])):
            text = f"prefers {genre_label(pair[0])} and {genre_label(pair[1])} films"
            claims.append({"claim": text, "type": "genre", "confidence": 0.0})
            if len(claims) >= max(0, claims_per_user - noise_claims):
                break

    if years and len(claims) < max(0, claims_per_user - noise_claims):
        median_year = float(np.median(years))
        if median_year < 1985:
            claims.append({"claim": "enjoys classic older movies", "type": "preference", "confidence": 0.0})
        elif median_year > 1995:
            claims.append({"claim": "prefers contemporary movies", "type": "preference", "confidence": 0.0})

    absent_noise = []
    recent_genres = set(genre_counts)
    for claim, claim_type in PROXY_UNSUPPORTED_CLAIMS:
        claim_genres = set(genre_words_for_claim(claim))
        if not claim_genres.intersection(recent_genres):
            absent_noise.append({"claim": claim, "type": claim_type, "confidence": 0.0})
    if not absent_noise:
        absent_noise = [{"claim": claim, "type": claim_type, "confidence": 0.0} for claim, claim_type in PROXY_UNSUPPORTED_CLAIMS]
    for idx in range(max(0, noise_claims)):
        claims.append(absent_noise[(uid + idx) % len(absent_noise)])

    fallback = [{"claim": "likes popular mainstream movies", "type": "preference", "confidence": 0.0}]
    while len(claims) < claims_per_user:
        claims.extend(fallback)
    return claims[:claims_per_user]


def generate_profiles(
    users: Sequence[int],
    ordered_histories: Sequence[np.ndarray],
    meta: ItemMetadata,
    args: argparse.Namespace,
) -> Tuple[Dict[int, List[Dict[str, object]]], pd.DataFrame]:
    profiles: Dict[int, List[Dict[str, object]]] = {}
    cost_rows: List[Dict[str, object]] = []
    unique_users = sorted(set(int(uid) for uid in users))
    generator: Optional[ChatProfileGenerator] = None
    if args.profile_mode == "api":
        generator = ChatProfileGenerator(
            args.cache_dir or (args.outdir / "profile_api_cache"),
            args.provider,
            args.model,
            args.api_url,
            args.api_timeout,
            args.api_max_retries,
            args.api_max_output_tokens,
            args.prompt_variant,
        )

    for row, uid in enumerate(unique_users, start=1):
        if generator is None:
            profiles[uid] = metadata_proxy_claims(
                uid,
                ordered_histories,
                meta,
                args.history_limit,
                args.claims_per_user,
                args.proxy_noise_claims,
            )
            cost_rows.append({"uid": uid, "input_tokens": 0, "output_tokens": 0, "latency_seconds": 0.0})
        else:
            profiles[uid], cost = generator.generate_one(uid, ordered_histories, meta, args.history_limit, args.claims_per_user)
            cost_rows.append({"uid": uid, **cost})
        if row % 50 == 0 or row == len(unique_users):
            print(f"profile generation {row}/{len(unique_users)} users", flush=True)
    return profiles, pd.DataFrame(cost_rows)


def claim_item_similarity(claim: str, iid: int, meta: ItemMetadata) -> float:
    claim_genres = set(genre_words_for_claim(claim))
    item_genres = set(normalize_genre(g) for g in meta.genres[iid])
    if claim_genres:
        meta_score = len(claim_genres.intersection(item_genres)) / max(1, len(claim_genres))
    else:
        meta_score = 0.0
    claim_tokens = set(simple_tokens(claim))
    item_tokens = set(item_text_tokens(iid, meta))
    token_overlap = len(claim_tokens.intersection(item_tokens))
    token_score = token_overlap / max(1, min(len(claim_tokens), len(item_tokens)))
    semantic_score = 0.75 * meta_score + 0.25 * token_score
    auxiliary_scores = [
        score
        for score in (
            temporal_support_for_claim(claim, iid, meta),
            structural_support_for_claim(claim, iid, meta),
        )
        if score is not None
    ]
    if auxiliary_scores:
        evidence_score = max(auxiliary_scores)
        return float(max(semantic_score, 0.80 * evidence_score + 0.20 * semantic_score))
    return float(semantic_score)


def score_claim(
    uid: int,
    claim_id: int,
    claim_entry: Mapping[str, object],
    ordered_histories: Sequence[np.ndarray],
    meta: ItemMetadata,
    history_limit: int,
    support_threshold: float,
) -> ClaimRecord:
    claim = str(claim_entry.get("claim", "")).strip()
    recent = ordered_histories[uid][-history_limit:]
    scored = [(int(iid), claim_item_similarity(claim, int(iid), meta)) for iid in recent]
    supporting = [iid for iid, score in scored if score >= support_threshold]
    max_score = max((score for _, score in scored), default=0.0)
    mean_supported_score = float(np.mean([score for _, score in scored if score >= support_threshold])) if supporting else 0.0
    freq_score = min(1.0, len(supporting) / 3.0)
    support_score = 0.55 * mean_supported_score + 0.25 * max_score + 0.20 * freq_score
    support_weight = sigmoid(8.0 * (support_score - 0.35))
    if len(supporting) == 0:
        status = "unsupported"
    elif len(supporting) == 1:
        status = "weakly_supported"
    else:
        status = "supported"
    return ClaimRecord(
        uid=uid,
        claim_id=claim_id,
        claim=claim,
        claim_type=str(claim_entry.get("type", "preference")),
        confidence=float(claim_entry.get("confidence", 0.0) or 0.0),
        support_count=len(supporting),
        support_score=float(support_score),
        support_weight=float(support_weight),
        status=status,
        supporting_items=supporting,
    )


def score_all_claims(
    profiles: Mapping[int, List[Dict[str, object]]],
    ordered_histories: Sequence[np.ndarray],
    meta: ItemMetadata,
    history_limit: int,
    support_threshold: float,
) -> Dict[int, List[ClaimRecord]]:
    out: Dict[int, List[ClaimRecord]] = {}
    for uid, claims in profiles.items():
        out[int(uid)] = [
            score_claim(int(uid), idx, claim, ordered_histories, meta, history_limit, support_threshold)
            for idx, claim in enumerate(claims)
        ]
    return out


def claim_records_to_jsonl(path: Path, records_by_user: Mapping[int, List[ClaimRecord]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for uid in sorted(records_by_user):
            for record in records_by_user[uid]:
                handle.write(json.dumps(record.__dict__, ensure_ascii=False) + "\n")


def raw_profiles_to_jsonl(path: Path, profiles: Mapping[int, List[Dict[str, object]]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for uid in sorted(profiles):
            handle.write(json.dumps({"uid": uid, "claims": profiles[uid]}, ensure_ascii=False) + "\n")


def repaired_profiles_to_jsonl(path: Path, records_by_user: Mapping[int, List[ClaimRecord]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for uid in sorted(records_by_user):
            records = records_by_user[uid]
            remove_claims = [
                {"claim": r.claim, "type": r.claim_type, "weight": 1.0}
                for r in records
                if r.status != "unsupported"
            ]
            weighted_claims = [
                {"claim": r.claim, "type": r.claim_type, "weight": r.support_weight, "status": r.status}
                for r in records
            ]
            handle.write(
                json.dumps(
                    {
                        "uid": uid,
                        "remove_repair": remove_claims,
                        "evidence_weighted_repair": weighted_claims,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def vector_from_history(uid: int, ordered_histories: Sequence[np.ndarray], meta: ItemMetadata, history_limit: int) -> Dict[str, float]:
    vector: Dict[str, float] = {}
    recent = ordered_histories[uid][-history_limit:]
    if len(recent) == 0:
        return vector
    denom = float(len(recent))
    for pos, iid_np in enumerate(recent, start=1):
        iid = int(iid_np)
        weight = pos / denom
        for genre in meta.genres[iid]:
            key = f"genre:{normalize_genre(genre)}"
            vector[key] = vector.get(key, 0.0) + weight
        for token in item_text_tokens(iid, meta):
            key = f"token:{token}"
            vector[key] = vector.get(key, 0.0) + 0.25 * weight
    return vector


def vector_from_claims(records: Sequence[ClaimRecord], mode: str) -> Dict[str, float]:
    vector: Dict[str, float] = {}
    for record in records:
        if mode == "remove" and record.status == "unsupported":
            continue
        if mode == "weighted":
            weight = record.support_weight
        else:
            weight = 1.0
        if weight <= 0.0:
            continue
        for genre in genre_words_for_claim(record.claim):
            key = f"genre:{genre}"
            vector[key] = vector.get(key, 0.0) + weight
        for token in simple_tokens(record.claim):
            key = f"token:{token}"
            vector[key] = vector.get(key, 0.0) + 0.25 * weight
    return vector


def evidence_coverage(uid: int, records: Sequence[ClaimRecord], ordered_histories: Sequence[np.ndarray], history_limit: int) -> float:
    recent = set(int(iid) for iid in ordered_histories[uid][-history_limit:])
    if not recent:
        return 0.0
    covered = set()
    for record in records:
        if record.status != "unsupported":
            covered.update(record.supporting_items)
    return len(covered.intersection(recent)) / len(recent)


def profile_faithfulness(
    records_by_user: Mapping[int, List[ClaimRecord]],
    ordered_histories: Sequence[np.ndarray],
    meta: ItemMetadata,
    history_limit: int,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    modes = [
        ("Raw Profile", "raw"),
        ("Remove Repair", "remove"),
        ("Evidence-Weighted Repair", "weighted"),
    ]
    for label, mode in modes:
        total_claims = 0
        unsupported = 0
        weighted_unsupported = 0.0
        total_weight = 0.0
        coverage_values: List[float] = []
        drift_values: List[float] = []
        retained_claims = 0
        for uid, records in records_by_user.items():
            history_vec = vector_from_history(uid, ordered_histories, meta, history_limit)
            claim_vec = vector_from_claims(records, mode)
            drift_values.append(1.0 - cosine_sparse(claim_vec, history_vec))
            coverage_values.append(evidence_coverage(uid, records, ordered_histories, history_limit))
            for record in records:
                if mode == "remove" and record.status == "unsupported":
                    continue
                weight = record.support_weight if mode == "weighted" else 1.0
                total_claims += 1
                retained_claims += 1
                total_weight += weight
                if record.status == "unsupported":
                    unsupported += 1
                    weighted_unsupported += weight
        rows.append(
            {
                "Method": label,
                "Claims": retained_claims,
                "UCR": unsupported / total_claims if total_claims else 0.0,
                "WeightedUCR": weighted_unsupported / total_weight if total_weight > 0.0 else 0.0,
                "EvidenceCoverage": float(np.mean(coverage_values)) if coverage_values else 0.0,
                "ProfileDriftScore": float(np.mean(drift_values)) if drift_values else 1.0,
            }
        )
    return pd.DataFrame(rows)


def profile_scores_for_batch(
    batch: CandidateBatch,
    records_by_user: Mapping[int, List[ClaimRecord]],
    meta: ItemMetadata,
    method: str,
) -> np.ndarray:
    scores = np.zeros_like(batch.scores, dtype=np.float32)
    for row, uid_np in enumerate(batch.users):
        uid = int(uid_np)
        records = records_by_user.get(uid, [])
        weighted_records: List[Tuple[ClaimRecord, float]] = []
        for record in records:
            if method == "raw":
                weight = 1.0
            elif method == "remove":
                weight = 0.0 if record.status == "unsupported" else 1.0
            elif method == "weighted":
                weight = record.support_weight
            else:
                raise ValueError(f"Unknown profile method: {method}")
            if weight > 0.0:
                weighted_records.append((record, weight))
        if not weighted_records:
            continue
        denom = sum(weight for _, weight in weighted_records)
        for pos, iid_np in enumerate(batch.candidates[row]):
            iid = int(iid_np)
            total = 0.0
            for record, weight in weighted_records:
                total += weight * claim_item_similarity(record.claim, iid, meta)
            scores[row, pos] = float(total / max(denom, 1e-8))
    return scores


def ranked_from_profile_scores(batch: CandidateBatch, profile_scores: np.ndarray, lam: float, topk: int) -> np.ndarray:
    ranked = np.zeros((len(batch.users), topk), dtype=np.int64)
    for row in range(len(batch.users)):
        base_z = zscore(batch.scores[row])
        profile_z = zscore(profile_scores[row])
        final_scores = base_z + float(lam) * profile_z
        order = topk_indices(final_scores.astype(np.float32), topk)
        ranked[row] = batch.candidates[row, order]
    return ranked


def metrics_from_ranked(ranked_topk: np.ndarray, targets: np.ndarray) -> pd.DataFrame:
    rows = []
    for row_idx, target_np in enumerate(targets):
        target = int(target_np)
        positions = np.flatnonzero(ranked_topk[row_idx] == target)
        hit = len(positions) > 0
        ndcg = 1.0 / math.log2(int(positions[0]) + 2) if hit else 0.0
        rows.append({"NDCG@20": ndcg, "Recall@20": float(hit), "HitRate@20": float(hit)})
    return pd.DataFrame(rows)


def metric_summary(metrics: pd.DataFrame) -> Dict[str, float]:
    return {col: float(metrics[col].mean()) for col in ["NDCG@20", "Recall@20", "HitRate@20"]}


def reliability_summary(method: str, base_metrics: pd.DataFrame, method_metrics: pd.DataFrame) -> Dict[str, object]:
    delta = method_metrics["NDCG@20"].to_numpy(np.float64) - base_metrics["NDCG@20"].to_numpy(np.float64)
    pos = delta[delta > 0.0]
    neg = delta[delta < 0.0]
    return {
        "Method": method,
        "HarmRate": float(np.mean(delta < 0.0)),
        "PositiveGainRate": float(np.mean(delta > 0.0)),
        "MeanDeltaNDCG@20": float(np.mean(delta)),
        "PositiveGainSum": float(pos.sum()) if len(pos) else 0.0,
        "NegativeGainSum": float(neg.sum()) if len(neg) else 0.0,
        "GainHarmRatio": float(pos.sum() / abs(neg.sum())) if len(neg) and abs(neg.sum()) > 1e-12 else np.inf,
    }


def select_lambda(
    method_label: str,
    method_key: str,
    val_batch: CandidateBatch,
    val_profile_scores: np.ndarray,
    lambda_grid: Sequence[float],
    topk: int,
) -> Tuple[float, pd.DataFrame]:
    rows: List[Dict[str, object]] = []
    for lam in lambda_grid:
        ranked = ranked_from_profile_scores(val_batch, val_profile_scores, lam, topk)
        metrics = metrics_from_ranked(ranked, val_batch.targets)
        summary = metric_summary(metrics)
        rows.append({"Method": method_label, "method_key": method_key, "lambda": float(lam), **summary})
    table = pd.DataFrame(rows)
    selected = table.sort_values(["NDCG@20", "Recall@20", "lambda"], ascending=[False, False, True]).iloc[0]
    return float(selected["lambda"]), table


def evaluate_methods(
    val_batch: CandidateBatch,
    test_batch: CandidateBatch,
    records_by_user: Mapping[int, List[ClaimRecord]],
    meta: ItemMetadata,
    lambda_grid: Sequence[float],
    topk: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame]]:
    base_ranked = test_batch.candidates[:, :topk]
    base_metrics = metrics_from_ranked(base_ranked, test_batch.targets)
    performance_rows = [{"Method": "LightGCN", "SelectedLambda": 0.0, **metric_summary(base_metrics)}]
    reliability_rows = [
        {
            "Method": "LightGCN",
            "HarmRate": 0.0,
            "PositiveGainRate": 0.0,
            "MeanDeltaNDCG@20": 0.0,
            "PositiveGainSum": 0.0,
            "NegativeGainSum": 0.0,
            "GainHarmRatio": np.nan,
        }
    ]
    lambda_tables: List[pd.DataFrame] = []
    per_user: Dict[str, pd.DataFrame] = {"LightGCN": base_metrics.copy()}

    methods = [
        ("LightGCN + Raw Profile", "raw"),
        ("LightGCN + Remove Repair", "remove"),
        ("LightGCN + Evidence-Weighted Repair", "weighted"),
    ]
    for label, method_key in methods:
        val_profile_scores = profile_scores_for_batch(val_batch, records_by_user, meta, method_key)
        test_profile_scores = profile_scores_for_batch(test_batch, records_by_user, meta, method_key)
        selected_lambda, lambda_table = select_lambda(
            label, method_key, val_batch, val_profile_scores, lambda_grid, topk
        )
        lambda_tables.append(lambda_table)
        ranked = ranked_from_profile_scores(test_batch, test_profile_scores, selected_lambda, topk)
        metrics = metrics_from_ranked(ranked, test_batch.targets)
        per_user[label] = metrics.copy()
        performance_rows.append({"Method": label, "SelectedLambda": selected_lambda, **metric_summary(metrics)})
        reliability_rows.append(reliability_summary(label, base_metrics, metrics))

    return pd.DataFrame(performance_rows), pd.DataFrame(reliability_rows), pd.concat(lambda_tables, ignore_index=True), per_user


def build_go_no_go(
    profile_mode: str,
    faithfulness: pd.DataFrame,
    performance: pd.DataFrame,
    reliability: pd.DataFrame,
) -> Dict[str, object]:
    raw_ucr = float(faithfulness.loc[faithfulness["Method"] == "Raw Profile", "UCR"].iloc[0])
    remove_ucr = float(faithfulness.loc[faithfulness["Method"] == "Remove Repair", "UCR"].iloc[0])
    egpr_weighted_ucr = float(
        faithfulness.loc[faithfulness["Method"] == "Evidence-Weighted Repair", "WeightedUCR"].iloc[0]
    )
    raw_harm = float(reliability.loc[reliability["Method"] == "LightGCN + Raw Profile", "HarmRate"].iloc[0])
    egpr_harm = float(
        reliability.loc[reliability["Method"] == "LightGCN + Evidence-Weighted Repair", "HarmRate"].iloc[0]
    )
    raw_ndcg = float(performance.loc[performance["Method"] == "LightGCN + Raw Profile", "NDCG@20"].iloc[0])
    egpr_ndcg = float(
        performance.loc[performance["Method"] == "LightGCN + Evidence-Weighted Repair", "NDCG@20"].iloc[0]
    )
    raw_ghr = float(reliability.loc[reliability["Method"] == "LightGCN + Raw Profile", "GainHarmRatio"].iloc[0])
    egpr_ghr = float(
        reliability.loc[reliability["Method"] == "LightGCN + Evidence-Weighted Repair", "GainHarmRatio"].iloc[0]
    )
    criteria = {
        "raw_ucr_ge_15pct": raw_ucr >= 0.15,
        "remove_ucr_relative_drop_ge_30pct": ((raw_ucr - remove_ucr) / raw_ucr >= 0.30) if raw_ucr > 0 else False,
        "egpr_weighted_ucr_relative_drop_ge_30pct": ((raw_ucr - egpr_weighted_ucr) / raw_ucr >= 0.30)
        if raw_ucr > 0
        else False,
        "raw_harm_ge_25pct": raw_harm >= 0.25,
        "egpr_harm_below_raw": egpr_harm < raw_harm,
        "egpr_ndcg_ge_or_close_to_raw": egpr_ndcg + 0.001 >= raw_ndcg,
        "egpr_ghr_above_raw": egpr_ghr > raw_ghr,
    }
    pass_count = sum(bool(v) for v in criteria.values())
    critical_signal_present = bool(criteria["raw_ucr_ge_15pct"] and criteria["raw_harm_ge_25pct"])
    decision = "go" if pass_count >= 4 and critical_signal_present else "no_go_or_revise"
    if profile_mode != "api":
        decision = "proxy_only_not_valid_for_deepseek_go_no_go"
    return {
        **criteria,
        "critical_signal_present": critical_signal_present,
        "pass_count": pass_count,
        "decision": decision,
        "raw_ucr": raw_ucr,
        "remove_ucr": remove_ucr,
        "egpr_weighted_ucr": egpr_weighted_ucr,
        "raw_harm": raw_harm,
        "egpr_harm": egpr_harm,
        "raw_ndcg": raw_ndcg,
        "egpr_ndcg": egpr_ndcg,
        "raw_ghr": raw_ghr,
        "egpr_ghr": egpr_ghr,
    }


def write_report(
    outdir: Path,
    args: argparse.Namespace,
    faithfulness: pd.DataFrame,
    performance: pd.DataFrame,
    reliability: pd.DataFrame,
    go_no_go: Mapping[str, object],
) -> None:
    def markdown_table(frame: pd.DataFrame) -> str:
        display = frame.copy()
        for col in display.columns:
            if pd.api.types.is_float_dtype(display[col]):
                display[col] = display[col].map(lambda value: "inf" if np.isinf(value) else f"{value:.6f}")
            else:
                display[col] = display[col].map(lambda value: "" if pd.isna(value) else str(value))
        headers = [str(col) for col in display.columns]
        rows = display.values.tolist()
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
        ]
        for row in rows:
            lines.append("| " + " | ".join(str(value) for value in row) + " |")
        return "\n".join(lines)

    lines = [
        "# EGPR ML-1M Mini-Pilot",
        "",
        f"Profile mode: `{args.profile_mode}`.",
        f"Dataset: ML-1M. Baseline: LightGCN. Candidate set: top-{args.top_candidates}. Output: top-{args.topk}.",
        f"Users per split: {args.max_users}. History limit: {args.history_limit}. Claims per user: {args.claims_per_user}. Prompt variant: {args.prompt_variant}.",
        "",
    ]
    if args.profile_mode != "api":
        lines.extend(
            [
                "**Important:** this run used a deterministic metadata proxy with optional injected unsupported claims.",
                "It validates the EGPR pipeline but is not evidence for DeepSeek profile behavior.",
                "",
            ]
        )
    lines.extend(
        [
            "## Profile Faithfulness",
            "",
            markdown_table(faithfulness),
            "",
            "## Recommendation Performance",
            "",
            markdown_table(performance),
            "",
            "## Reliability",
            "",
            markdown_table(reliability),
            "",
            "## Go / No-Go",
            "",
            "```json",
            json.dumps(go_no_go, indent=2),
            "```",
            "",
            "## Artifacts",
            "",
            "- `user_history.jsonl`",
            "- `raw_profiles.jsonl`",
            "- `claim_support.jsonl`",
            "- `repaired_profiles.jsonl`",
            "- `table1_profile_faithfulness.csv`",
            "- `table2_recommendation_performance.csv`",
            "- `table3_reliability.csv`",
            "- `table4_lambda_validation.csv`",
            "- `go_no_go.json`",
            "- `run_manifest.json`",
            "",
        ]
    )
    (outdir / "egpr_pilot_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    ensure_dirs(args.outdir)
    set_seed(args.seed)
    train, val, test, _ = read_interaction_split(args.datadir)
    n_users, n_items = infer_shape(train, val, test)
    train_histories = build_user_histories(train, n_users)
    ordered_train_histories = build_ordered_histories(train, n_users)
    val_exclude = build_exclude_lists(train_histories, None, n_users)
    test_exclude = build_exclude_lists(train_histories, val, n_users)
    meta = read_item_metadata(args.movies_path, args.datadir / "mappings.json", n_items)

    val_eval = sample_eval_frame(val, args.max_users, args.seed + 1)
    test_eval = sample_eval_frame(test, args.max_users, args.seed + 2)
    profile_users = sorted(set(val_eval["uid"].astype(int).tolist() + test_eval["uid"].astype(int).tolist()))
    write_user_histories(args.outdir, profile_users, ordered_train_histories, meta, args.history_limit)

    device = device_from_arg(args.device)
    print("loading LightGCN embeddings", flush=True)
    user_emb, item_emb = load_lightgcn_embeddings(args.config, args.lightgcn_ckpt, train, n_users, n_items, device)
    print("generating LightGCN top candidates", flush=True)
    val_batch = generate_lightgcn_candidates(
        val_eval, val_exclude, user_emb, item_emb, args.top_candidates, args.eval_batch_size, "val"
    )
    test_batch = generate_lightgcn_candidates(
        test_eval, test_exclude, user_emb, item_emb, args.top_candidates, args.eval_batch_size, "test"
    )
    save_candidates(args.outdir, val_batch)
    save_candidates(args.outdir, test_batch)

    profiles, profile_cost = generate_profiles(profile_users, ordered_train_histories, meta, args)
    raw_profiles_to_jsonl(args.outdir / "raw_profiles.jsonl", profiles)
    profile_cost.to_csv(args.outdir / "profile_cost_trace.csv", index=False)

    claim_records = score_all_claims(
        profiles, ordered_train_histories, meta, args.history_limit, args.support_threshold
    )
    claim_records_to_jsonl(args.outdir / "claim_support.jsonl", claim_records)
    repaired_profiles_to_jsonl(args.outdir / "repaired_profiles.jsonl", claim_records)

    faithfulness = profile_faithfulness(claim_records, ordered_train_histories, meta, args.history_limit)
    performance, reliability, lambda_table, per_user_metrics = evaluate_methods(
        val_batch, test_batch, claim_records, meta, args.lambda_grid, args.topk
    )
    go_no_go = build_go_no_go(args.profile_mode, faithfulness, performance, reliability)

    faithfulness.to_csv(args.outdir / "table1_profile_faithfulness.csv", index=False)
    performance.to_csv(args.outdir / "table2_recommendation_performance.csv", index=False)
    reliability.to_csv(args.outdir / "table3_reliability.csv", index=False)
    lambda_table.to_csv(args.outdir / "table4_lambda_validation.csv", index=False)
    for method_name, metrics in per_user_metrics.items():
        safe_name = re.sub(r"[^A-Za-z0-9]+", "_", method_name).strip("_").lower()
        per_user_frame = metrics.copy()
        per_user_frame.insert(0, "target", test_batch.targets.astype(int))
        per_user_frame.insert(0, "uid", test_batch.users.astype(int))
        per_user_frame.to_csv(args.outdir / f"per_user_{safe_name}.csv", index=False)
    (args.outdir / "go_no_go.json").write_text(json.dumps(go_no_go, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "status": "completed",
        "profile_mode": args.profile_mode,
        "provider": args.provider if args.profile_mode == "api" else None,
        "model": args.model if args.profile_mode == "api" else None,
        "seed": args.seed,
        "datadir": str(args.datadir),
        "movies_path": str(args.movies_path),
        "max_users": args.max_users,
        "top_candidates": args.top_candidates,
        "topk": args.topk,
        "history_limit": args.history_limit,
        "claims_per_user": args.claims_per_user,
        "prompt_variant": args.prompt_variant,
        "lambda_grid": args.lambda_grid,
        "support_threshold": args.support_threshold,
        "proxy_noise_claims": args.proxy_noise_claims if args.profile_mode == "metadata_proxy" else None,
        "go_no_go": go_no_go,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (args.outdir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    write_report(args.outdir, args, faithfulness, performance, reliability, go_no_go)
    print(f"Done. Report: {args.outdir / 'egpr_pilot_report.md'}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Claim injection test for the ML-1M EGPR pilot.

This script reuses cached DeepSeek raw profiles and LightGCN candidate sets from
an EGPR profile-repair run. It appends unsupported preference claims chosen to
avoid each user's recent-history genres, then tests whether injected claims cause
recommendation harm and whether evidence-grounded repair removes or downweights
the damage.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path('/root/temporal_popularity_pilot')
SRC = ROOT / 'src'
SCRIPT_DIR = ROOT / 'scripts'
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SCRIPT_DIR))

from temporal_popularity.data import read_interaction_split, infer_shape, set_seed  # noqa: E402
from run_egpr_profile_repair_pilot import (  # noqa: E402
    CandidateBatch,
    ClaimRecord,
    claim_records_to_jsonl,
    cosine_sparse,
    evidence_coverage,
    genre_words_for_claim,
    metrics_from_ranked,
    profile_scores_for_batch,
    ranked_from_profile_scores,
    read_item_metadata,
    reliability_summary,
    score_all_claims,
    vector_from_claims,
    vector_from_history,
)
from run_egpr_profile_repair_pilot import build_ordered_histories  # noqa: E402


@dataclass(frozen=True)
class InjectionClaim:
    claim: str
    claim_type: str
    target_genres: Tuple[str, ...]


INJECTION_POOL: Tuple[InjectionClaim, ...] = (
    InjectionClaim('likes science fiction thrillers', 'theme', ('sci-fi', 'thriller')),
    InjectionClaim('prefers dark crime dramas', 'theme', ('crime', 'drama')),
    InjectionClaim('enjoys romantic historical films', 'theme', ('romance', 'drama')),
    InjectionClaim('likes superhero action movies', 'theme', ('action',)),
    InjectionClaim('prefers animated family movies', 'genre', ('animation', "children's")),
    InjectionClaim('enjoys horror and supernatural suspense', 'theme', ('horror', 'thriller')),
    InjectionClaim('likes musical performance stories', 'style', ('musical',)),
    InjectionClaim('prefers western frontier adventures', 'genre', ('western', 'adventure')),
    InjectionClaim('enjoys war dramas', 'genre', ('war', 'drama')),
    InjectionClaim('prefers fantasy adventure films', 'genre', ('fantasy', 'adventure')),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-run', type=Path, default=ROOT / 'results/egpr_profile_repair/ml1m_seed42_deepseek_500')
    parser.add_argument('--outdir', type=Path, default=ROOT / 'results/egpr_profile_repair/ml1m_seed42_deepseek_500_injection')
    parser.add_argument('--datadir', type=Path, default=ROOT / 'data/ml1m')
    parser.add_argument(
        '--movies-path',
        type=Path,
        default=Path('/root/autodl-tmp/projects_archive/MGPrompt-CDR/data/raw/ml-1m/movies.dat'),
    )
    parser.add_argument('--history-limit', type=int, default=20)
    parser.add_argument('--topk', type=int, default=20)
    parser.add_argument('--inject-claims', type=int, default=2)
    parser.add_argument('--lambda-grid', nargs='+', type=float, default=[0.05, 0.1, 0.2, 0.3, 0.5, 1.0])
    parser.add_argument('--support-threshold', type=float, default=0.25)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_genre(genre: str) -> str:
    return genre.strip().lower()


def load_profiles(path: Path) -> Dict[int, List[Dict[str, object]]]:
    profiles: Dict[int, List[Dict[str, object]]] = {}
    with path.open('r', encoding='utf-8') as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            profiles[int(row['uid'])] = list(row['claims'])
    return profiles


def load_batch(path: Path, split_name: str) -> CandidateBatch:
    data = np.load(path, allow_pickle=False)
    return CandidateBatch(
        users=data['users'].astype(np.int64),
        targets=data['targets'].astype(np.int64),
        candidates=data['candidates'].astype(np.int64),
        scores=data['scores'].astype(np.float32),
        split_name=split_name,
    )


def recent_history_genres(uid: int, ordered_histories: Sequence[np.ndarray], meta, history_limit: int) -> Tuple[str, ...]:
    genres = set()
    for iid_np in ordered_histories[uid][-history_limit:]:
        iid = int(iid_np)
        genres.update(normalize_genre(genre) for genre in meta.genres[iid])
    return tuple(sorted(genres))


def choose_injections(
    uid: int,
    history_genres: Iterable[str],
    inject_count: int,
) -> List[InjectionClaim]:
    genre_set = set(history_genres)
    scored: List[Tuple[int, int, InjectionClaim]] = []
    for idx, injection in enumerate(INJECTION_POOL):
        overlap = len(set(injection.target_genres).intersection(genre_set))
        scored.append((overlap, (uid + idx) % len(INJECTION_POOL), injection))
    # Prefer zero-overlap claims; second key gives deterministic user-level variety.
    scored.sort(key=lambda row: (row[0], row[1], row[2].claim))
    chosen: List[InjectionClaim] = []
    seen = set()
    for _, _, injection in scored:
        if injection.claim in seen:
            continue
        chosen.append(injection)
        seen.add(injection.claim)
        if len(chosen) >= inject_count:
            break
    return chosen


def inject_profiles(
    raw_profiles: Mapping[int, List[Dict[str, object]]],
    users: Sequence[int],
    ordered_histories: Sequence[np.ndarray],
    meta,
    history_limit: int,
    inject_count: int,
) -> Tuple[Dict[int, List[Dict[str, object]]], Dict[int, List[InjectionClaim]]]:
    injected_profiles: Dict[int, List[Dict[str, object]]] = {int(uid): [dict(c) for c in claims] for uid, claims in raw_profiles.items()}
    injected_claims: Dict[int, List[InjectionClaim]] = {}
    for uid_np in sorted(set(int(uid) for uid in users)):
        genres = recent_history_genres(uid_np, ordered_histories, meta, history_limit)
        chosen = choose_injections(uid_np, genres, inject_count)
        injected_claims[uid_np] = chosen
        injected_profiles.setdefault(uid_np, [])
        for injection in chosen:
            injected_profiles[uid_np].append(
                {'claim': injection.claim, 'type': injection.claim_type, 'confidence': 0.0, 'injected': True}
            )
    return injected_profiles, injected_claims


def write_injected_profiles(path: Path, profiles: Mapping[int, List[Dict[str, object]]]) -> None:
    with path.open('w', encoding='utf-8') as handle:
        for uid in sorted(profiles):
            handle.write(json.dumps({'uid': uid, 'claims': profiles[uid]}, ensure_ascii=False) + '\n')


def write_injected_claims(path: Path, injected_claims: Mapping[int, List[InjectionClaim]]) -> None:
    with path.open('w', encoding='utf-8') as handle:
        for uid in sorted(injected_claims):
            rows = [
                {'claim': c.claim, 'type': c.claim_type, 'target_genres': list(c.target_genres)}
                for c in injected_claims[uid]
            ]
            handle.write(json.dumps({'uid': uid, 'injected_claims': rows}, ensure_ascii=False) + '\n')


def metric_summary(metrics: pd.DataFrame) -> Dict[str, float]:
    return {col: float(metrics[col].mean()) for col in ['NDCG@20', 'Recall@20', 'HitRate@20']}


def select_lambda(
    label: str,
    method_key: str,
    val_batch: CandidateBatch,
    records_by_user: Mapping[int, List[ClaimRecord]],
    meta,
    lambda_grid: Sequence[float],
    topk: int,
) -> Tuple[float, pd.DataFrame]:
    profile_scores = profile_scores_for_batch(val_batch, records_by_user, meta, method_key)
    rows: List[Dict[str, object]] = []
    for lam in lambda_grid:
        ranked = ranked_from_profile_scores(val_batch, profile_scores, float(lam), topk)
        metrics = metrics_from_ranked(ranked, val_batch.targets)
        rows.append({'Method': label, 'method_key': method_key, 'lambda': float(lam), **metric_summary(metrics)})
    table = pd.DataFrame(rows)
    selected = table.sort_values(['NDCG@20', 'Recall@20', 'lambda'], ascending=[False, False, True]).iloc[0]
    return float(selected['lambda']), table


def rank_test_method(
    test_batch: CandidateBatch,
    records_by_user: Mapping[int, List[ClaimRecord]],
    meta,
    method_key: str,
    lam: float,
    topk: int,
) -> Tuple[np.ndarray, pd.DataFrame]:
    profile_scores = profile_scores_for_batch(test_batch, records_by_user, meta, method_key)
    ranked = ranked_from_profile_scores(test_batch, profile_scores, lam, topk)
    metrics = metrics_from_ranked(ranked, test_batch.targets)
    return ranked, metrics


def faithfulness_row(
    label: str,
    records_by_user: Mapping[int, List[ClaimRecord]],
    ordered_histories: Sequence[np.ndarray],
    meta,
    history_limit: int,
    profile_mode: str,
) -> Dict[str, object]:
    total_claims = 0
    unsupported = 0
    total_weight = 0.0
    unsupported_weight = 0.0
    coverage_values: List[float] = []
    drift_values: List[float] = []
    for uid, records in records_by_user.items():
        history_vec = vector_from_history(uid, ordered_histories, meta, history_limit)
        claim_vec = vector_from_claims(records, profile_mode)
        drift_values.append(1.0 - cosine_sparse(claim_vec, history_vec))
        coverage_values.append(evidence_coverage(uid, records, ordered_histories, history_limit))
        for record in records:
            if profile_mode == 'remove' and record.status == 'unsupported':
                continue
            weight = record.support_weight if profile_mode == 'weighted' else 1.0
            total_claims += 1
            total_weight += weight
            if record.status == 'unsupported':
                unsupported += 1
                unsupported_weight += weight
    return {
        'Method': label,
        'Claims': total_claims,
        'UCR': unsupported / total_claims if total_claims else 0.0,
        'WeightedUCR': unsupported_weight / total_weight if total_weight > 0.0 else 0.0,
        'EvidenceCoverage': float(np.mean(coverage_values)) if coverage_values else 0.0,
        'ProfileDriftScore': float(np.mean(drift_values)) if drift_values else 1.0,
    }


def exposure_to_injected_genres(
    ranked: np.ndarray,
    users: np.ndarray,
    injected_claims: Mapping[int, List[InjectionClaim]],
    meta,
) -> np.ndarray:
    exposures = np.zeros(len(users), dtype=np.float64)
    for row, uid_np in enumerate(users):
        uid = int(uid_np)
        target_genres = set()
        for claim in injected_claims.get(uid, []):
            target_genres.update(claim.target_genres)
        if not target_genres:
            continue
        hits = 0
        for iid_np in ranked[row]:
            item_genres = set(normalize_genre(genre) for genre in meta.genres[int(iid_np)])
            hits += int(bool(target_genres.intersection(item_genres)))
        exposures[row] = hits / max(1, ranked.shape[1])
    return exposures


def injected_claim_support_summary(
    records_by_user: Mapping[int, List[ClaimRecord]],
    raw_lengths: Mapping[int, int],
) -> Dict[str, float]:
    statuses = []
    for uid, records in records_by_user.items():
        start = raw_lengths.get(uid, 0)
        statuses.extend(record.status for record in records[start:])
    if not statuses:
        return {'InjectedClaims': 0, 'InjectedUnsupportedRate': 0.0, 'InjectedSupportedRate': 0.0}
    return {
        'InjectedClaims': len(statuses),
        'InjectedUnsupportedRate': float(np.mean([status == 'unsupported' for status in statuses])),
        'InjectedSupportedRate': float(np.mean([status == 'supported' for status in statuses])),
    }


def markdown_table(frame: pd.DataFrame) -> str:
    display = frame.copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(lambda value: 'inf' if np.isinf(value) else f'{value:.6f}')
        else:
            display[col] = display[col].map(lambda value: '' if pd.isna(value) else str(value))
    headers = [str(col) for col in display.columns]
    lines = ['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join('---' for _ in headers) + ' |']
    for row in display.values.tolist():
        lines.append('| ' + ' | '.join(str(value) for value in row) + ' |')
    return '\n'.join(lines)


def write_report(
    outdir: Path,
    args: argparse.Namespace,
    faithfulness: pd.DataFrame,
    performance: pd.DataFrame,
    reliability: pd.DataFrame,
    lambda_table: pd.DataFrame,
    decision: Mapping[str, object],
) -> None:
    lines = [
        '# EGPR Claim Injection Test',
        '',
        f'Base run: `{args.base_run}`.',
        f'Injected claims per user: {args.inject_claims}. Candidate set: top-{performance.attrs.get("top_candidates", "unknown")}. Output: top-{args.topk}.',
        '',
        '## Faithfulness / Drift',
        '',
        markdown_table(faithfulness),
        '',
        '## Recommendation Performance',
        '',
        markdown_table(performance),
        '',
        '## Reliability',
        '',
        markdown_table(reliability),
        '',
        '## Lambda Validation',
        '',
        markdown_table(lambda_table),
        '',
        '## Decision',
        '',
        '```json',
        json.dumps(decision, indent=2),
        '```',
        '',
        '## Artifacts',
        '',
        '- `injected_profiles.jsonl`',
        '- `injected_claims.jsonl`',
        '- `claim_support_raw.jsonl`',
        '- `claim_support_injected.jsonl`',
        '- `table1_injection_faithfulness.csv`',
        '- `table2_injection_performance.csv`',
        '- `table3_injection_reliability.csv`',
        '- `table4_injection_lambda_validation.csv`',
        '- `claim_injection_decision.json`',
        '',
    ]
    (outdir / 'claim_injection_report.md').write_text('\n'.join(lines), encoding='utf-8')


def main() -> None:
    args = parse_args()
    ensure_dir(args.outdir)
    set_seed(args.seed)

    train, val, test, _ = read_interaction_split(args.datadir)
    n_users, n_items = infer_shape(train, val, test)
    ordered_histories = build_ordered_histories(train, n_users)
    meta = read_item_metadata(args.movies_path, args.datadir / 'mappings.json', n_items)

    val_batch = load_batch(args.base_run / 'candidates_lightgcn_val.npz', 'val')
    test_batch = load_batch(args.base_run / 'candidates_lightgcn_test.npz', 'test')
    raw_profiles = load_profiles(args.base_run / 'raw_profiles.jsonl')
    profile_users = sorted(set(val_batch.users.astype(int).tolist() + test_batch.users.astype(int).tolist()))
    raw_profiles = {uid: raw_profiles[uid] for uid in profile_users if uid in raw_profiles}

    injected_profiles, injected_claims = inject_profiles(
        raw_profiles,
        profile_users,
        ordered_histories,
        meta,
        args.history_limit,
        args.inject_claims,
    )
    write_injected_profiles(args.outdir / 'injected_profiles.jsonl', injected_profiles)
    write_injected_claims(args.outdir / 'injected_claims.jsonl', injected_claims)

    raw_records = score_all_claims(raw_profiles, ordered_histories, meta, args.history_limit, args.support_threshold)
    injected_records = score_all_claims(injected_profiles, ordered_histories, meta, args.history_limit, args.support_threshold)
    claim_records_to_jsonl(args.outdir / 'claim_support_raw.jsonl', raw_records)
    claim_records_to_jsonl(args.outdir / 'claim_support_injected.jsonl', injected_records)

    raw_lengths = {uid: len(claims) for uid, claims in raw_profiles.items()}
    injection_support = injected_claim_support_summary(injected_records, raw_lengths)

    base_ranked = test_batch.candidates[:, : args.topk]
    base_metrics = metrics_from_ranked(base_ranked, test_batch.targets)
    base_summary = metric_summary(base_metrics)

    methods = [
        ('LightGCN + Raw Profile', raw_records, 'raw'),
        ('Injected Unsupported Profile', injected_records, 'raw'),
        ('EGPR after Injection (Remove)', injected_records, 'remove'),
        ('EGPR after Injection (Weighted)', injected_records, 'weighted'),
    ]

    performance_rows: List[Dict[str, object]] = [{'Method': 'LightGCN', 'SelectedLambda': 0.0, **base_summary}]
    reliability_rows: List[Dict[str, object]] = [{
        'Method': 'LightGCN',
        'HarmRate': 0.0,
        'PositiveGainRate': 0.0,
        'MeanDeltaNDCG@20': 0.0,
        'PositiveGainSum': 0.0,
        'NegativeGainSum': 0.0,
        'GainHarmRatio': np.nan,
        'InjectedGenreExposure@20': float(np.mean(exposure_to_injected_genres(base_ranked, test_batch.users, injected_claims, meta))),
        'ExposureDeltaVsRaw': np.nan,
    }]
    lambda_tables: List[pd.DataFrame] = []
    ranked_by_method: Dict[str, np.ndarray] = {'LightGCN': base_ranked}
    metrics_by_method: Dict[str, pd.DataFrame] = {'LightGCN': base_metrics}

    for label, records, method_key in methods:
        selected_lambda, lambda_table = select_lambda(label, method_key, val_batch, records, meta, args.lambda_grid, args.topk)
        lambda_tables.append(lambda_table)
        ranked, metrics = rank_test_method(test_batch, records, meta, method_key, selected_lambda, args.topk)
        ranked_by_method[label] = ranked
        metrics_by_method[label] = metrics
        performance_rows.append({'Method': label, 'SelectedLambda': selected_lambda, **metric_summary(metrics)})

    raw_exposure = exposure_to_injected_genres(
        ranked_by_method['LightGCN + Raw Profile'], test_batch.users, injected_claims, meta
    )
    for label in ['LightGCN + Raw Profile', 'Injected Unsupported Profile', 'EGPR after Injection (Remove)', 'EGPR after Injection (Weighted)']:
        reliability = reliability_summary(label, base_metrics, metrics_by_method[label])
        exposure = exposure_to_injected_genres(ranked_by_method[label], test_batch.users, injected_claims, meta)
        reliability['InjectedGenreExposure@20'] = float(np.mean(exposure))
        reliability['ExposureDeltaVsRaw'] = float(np.mean(exposure - raw_exposure))
        reliability_rows.append(reliability)

    faithfulness_rows = [
        faithfulness_row('Raw Profile', raw_records, ordered_histories, meta, args.history_limit, 'raw'),
        faithfulness_row('Injected Unsupported Profile', injected_records, ordered_histories, meta, args.history_limit, 'raw'),
        faithfulness_row('EGPR after Injection (Remove)', injected_records, ordered_histories, meta, args.history_limit, 'remove'),
        faithfulness_row('EGPR after Injection (Weighted)', injected_records, ordered_histories, meta, args.history_limit, 'weighted'),
    ]

    faithfulness = pd.DataFrame(faithfulness_rows)
    performance = pd.DataFrame(performance_rows)
    reliability = pd.DataFrame(reliability_rows)
    lambda_table = pd.concat(lambda_tables, ignore_index=True)

    raw_ndcg = float(performance.loc[performance['Method'] == 'LightGCN + Raw Profile', 'NDCG@20'].iloc[0])
    inj_ndcg = float(performance.loc[performance['Method'] == 'Injected Unsupported Profile', 'NDCG@20'].iloc[0])
    egpr_ndcg = float(performance.loc[performance['Method'] == 'EGPR after Injection (Weighted)', 'NDCG@20'].iloc[0])
    raw_harm = float(reliability.loc[reliability['Method'] == 'LightGCN + Raw Profile', 'HarmRate'].iloc[0])
    inj_harm = float(reliability.loc[reliability['Method'] == 'Injected Unsupported Profile', 'HarmRate'].iloc[0])
    egpr_harm = float(reliability.loc[reliability['Method'] == 'EGPR after Injection (Weighted)', 'HarmRate'].iloc[0])
    raw_exposure_mean = float(reliability.loc[reliability['Method'] == 'LightGCN + Raw Profile', 'InjectedGenreExposure@20'].iloc[0])
    inj_exposure_mean = float(reliability.loc[reliability['Method'] == 'Injected Unsupported Profile', 'InjectedGenreExposure@20'].iloc[0])
    egpr_exposure_mean = float(reliability.loc[reliability['Method'] == 'EGPR after Injection (Weighted)', 'InjectedGenreExposure@20'].iloc[0])

    decision = {
        **injection_support,
        'injected_ndcg_drop_vs_raw': inj_ndcg - raw_ndcg,
        'egpr_ndcg_recovery_vs_injected': egpr_ndcg - inj_ndcg,
        'injected_harm_delta_vs_raw': inj_harm - raw_harm,
        'egpr_harm_delta_vs_injected': egpr_harm - inj_harm,
        'injected_exposure_delta_vs_raw': inj_exposure_mean - raw_exposure_mean,
        'egpr_exposure_delta_vs_injected': egpr_exposure_mean - inj_exposure_mean,
        'injection_causes_harm': bool(inj_harm > raw_harm or inj_ndcg < raw_ndcg),
        'egpr_repairs_harm': bool(egpr_harm < inj_harm or egpr_ndcg > inj_ndcg),
        'egpr_salvage_signal': bool((inj_harm > raw_harm or inj_ndcg < raw_ndcg) and (egpr_harm < inj_harm or egpr_ndcg > inj_ndcg)),
        'updated_at_utc': datetime.now(timezone.utc).isoformat(),
    }

    faithfulness.to_csv(args.outdir / 'table1_injection_faithfulness.csv', index=False)
    performance.to_csv(args.outdir / 'table2_injection_performance.csv', index=False)
    reliability.to_csv(args.outdir / 'table3_injection_reliability.csv', index=False)
    lambda_table.to_csv(args.outdir / 'table4_injection_lambda_validation.csv', index=False)
    (args.outdir / 'claim_injection_decision.json').write_text(json.dumps(decision, indent=2) + '\n', encoding='utf-8')
    manifest = {
        'status': 'completed',
        'base_run': str(args.base_run),
        'inject_claims': args.inject_claims,
        'history_limit': args.history_limit,
        'topk': args.topk,
        'lambda_grid': args.lambda_grid,
        'support_threshold': args.support_threshold,
        'decision': decision,
        'updated_at_utc': datetime.now(timezone.utc).isoformat(),
    }
    (args.outdir / 'run_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    performance.attrs['top_candidates'] = int(test_batch.candidates.shape[1])
    write_report(args.outdir, args, faithfulness, performance, reliability, lambda_table, decision)
    print(f'Done. Report: {args.outdir / "claim_injection_report.md"}', flush=True)


if __name__ == '__main__':
    main()

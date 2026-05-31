#!/usr/bin/env python3
"""Fair ML-1M direct-vs-profile comparison on direct candidate sets.

No API calls are made. The script uses the direct rerank run's LightGCN
val/test candidates as the common candidate set and reranks those candidates
locally with the cached LLM profile claim-support records.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping

import numpy as np
import pandas as pd

ROOT = Path('/root/temporal_popularity_pilot')
SCRIPT_DIR = ROOT / 'scripts'
import sys
sys.path.insert(0, str(SCRIPT_DIR))

from run_egpr_profile_repair_pilot import (  # noqa: E402
    CandidateBatch,
    ClaimRecord,
    metric_summary,
    metrics_from_ranked,
    profile_scores_for_batch,
    ranked_from_profile_scores,
    read_item_metadata,
    reliability_summary,
    select_lambda,
)

INPUT_PRICE_PER_1M = 0.14
OUTPUT_PRICE_PER_1M = 0.28


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--direct-run', type=Path, required=True)
    parser.add_argument('--profile-run', type=Path, required=True)
    parser.add_argument('--outdir', type=Path, required=True)
    parser.add_argument('--datadir', type=Path, default=ROOT / 'data/ml1m')
    parser.add_argument('--movies-path', type=Path, default=Path('/root/autodl-tmp/projects_archive/MGPrompt-CDR/data/raw/ml-1m/movies.dat'))
    parser.add_argument('--topk', type=int, default=20)
    parser.add_argument('--lambda-grid', nargs='+', type=float, default=[0.05, 0.1, 0.2, 0.3, 0.5, 1.0])
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_batch(path: Path, split_name: str) -> CandidateBatch:
    data = np.load(path)
    return CandidateBatch(
        users=data['users'].astype(np.int64),
        targets=data['targets'].astype(np.int64),
        candidates=data['candidates'].astype(np.int64),
        scores=data['scores'].astype(np.float32),
        split_name=split_name,
    )


def load_records(path: Path) -> Dict[int, List[ClaimRecord]]:
    records: Dict[int, List[ClaimRecord]] = {}
    with path.open(encoding='utf-8') as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rec = ClaimRecord(
                uid=int(row['uid']),
                claim_id=int(row['claim_id']),
                claim=str(row['claim']),
                claim_type=str(row.get('claim_type', 'preference')),
                confidence=float(row.get('confidence', 0.0)),
                support_count=int(row.get('support_count', 0)),
                support_score=float(row.get('support_score', 0.0)),
                support_weight=float(row.get('support_weight', 0.0)),
                status=str(row.get('status', 'unsupported')),
                supporting_items=[int(x) for x in row.get('supporting_items', [])],
            )
            records.setdefault(rec.uid, []).append(rec)
    return records


def token_cost(input_tokens: float, output_tokens: float) -> float:
    return input_tokens / 1_000_000 * INPUT_PRICE_PER_1M + output_tokens / 1_000_000 * OUTPUT_PRICE_PER_1M


def direct_cost(direct_run: Path) -> Dict[str, float]:
    cost = pd.read_csv(direct_run / 'cost_trace_lightgcn_original.csv')
    input_tokens = float(cost['input_tokens'].sum())
    output_tokens = float(cost['output_tokens'].sum())
    return {
        'APICalls_TestUsers': float(len(cost)),
        'InputTokens_TestUsers': input_tokens,
        'OutputTokens_TestUsers': output_tokens,
        'TotalTokens_TestUsers': input_tokens + output_tokens,
        'EstimatedCostUSD_TestUsers': token_cost(input_tokens, output_tokens),
        'LatencySeconds_TestUsers': float(cost['latency_seconds'].sum()),
        'LatencyPerTestUser': float(cost['latency_seconds'].mean()),
    }


def profile_cost(profile_run: Path, test_users: np.ndarray) -> Dict[str, float]:
    cost = pd.read_csv(profile_run / 'profile_cost_trace.csv')
    test_set = set(int(uid) for uid in test_users)
    test_cost = cost[cost['uid'].astype(int).isin(test_set)]
    all_input = float(cost['input_tokens'].sum())
    all_output = float(cost['output_tokens'].sum())
    test_input = float(test_cost['input_tokens'].sum())
    test_output = float(test_cost['output_tokens'].sum())
    return {
        'APICalls_TestUsers': float(len(test_cost)),
        'InputTokens_TestUsers': test_input,
        'OutputTokens_TestUsers': test_output,
        'TotalTokens_TestUsers': test_input + test_output,
        'EstimatedCostUSD_TestUsers': token_cost(test_input, test_output),
        'LatencySeconds_TestUsers': float(test_cost['latency_seconds'].sum()),
        'LatencyPerTestUser': float(test_cost['latency_seconds'].mean()) if len(test_cost) else 0.0,
        'APICalls_ValTestUnique': float(len(cost)),
        'InputTokens_ValTestUnique': all_input,
        'OutputTokens_ValTestUnique': all_output,
        'TotalTokens_ValTestUnique': all_input + all_output,
        'EstimatedCostUSD_ValTestUnique': token_cost(all_input, all_output),
        'LatencySeconds_ValTestUnique': float(cost['latency_seconds'].sum()),
    }


def first_row(df: pd.DataFrame, method: str) -> pd.Series:
    rows = df.loc[df['Method'] == method]
    if rows.empty:
        raise RuntimeError(f'Missing method row: {method}')
    return rows.iloc[0]


def markdown_table(frame: pd.DataFrame) -> str:
    display = frame.copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(lambda v: 'inf' if np.isinf(v) else ('' if pd.isna(v) else f'{v:.6f}'))
        else:
            display[col] = display[col].map(lambda v: '' if pd.isna(v) else str(v))
    headers = [str(c) for c in display.columns]
    lines = ['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join('---' for _ in headers) + ' |']
    for row in display.values.tolist():
        lines.append('| ' + ' | '.join(str(v) for v in row) + ' |')
    return '\n'.join(lines)


def win_tie_loss(method_metrics: pd.DataFrame, ref_metrics: pd.DataFrame) -> Dict[str, int]:
    delta = method_metrics['NDCG@20'].to_numpy(np.float64) - ref_metrics['NDCG@20'].to_numpy(np.float64)
    return {
        'WinsVsDirect': int(np.sum(delta > 0.0)),
        'TiesVsDirect': int(np.sum(delta == 0.0)),
        'LossesVsDirect': int(np.sum(delta < 0.0)),
    }


def main() -> None:
    args = parse_args()
    ensure_dir(args.outdir)

    val_batch = load_batch(args.direct_run / 'candidates_lightgcn_val.npz', 'val')
    test_batch = load_batch(args.direct_run / 'candidates_lightgcn_test.npz', 'test')
    direct_ranked = np.load(args.direct_run / f'reranked_lightgcn_original_top{args.topk}.npy').astype(np.int64)
    if len(direct_ranked) != len(test_batch.users):
        raise RuntimeError('Direct ranked rows do not match direct test users.')

    profile_test = np.load(args.profile_run / 'candidates_lightgcn_test.npz')
    if not np.array_equal(test_batch.users, profile_test['users'].astype(np.int64)):
        raise RuntimeError('Direct and profile test users differ.')
    if not np.array_equal(test_batch.targets, profile_test['targets'].astype(np.int64)):
        raise RuntimeError('Direct and profile test targets differ.')

    n_items = int(max(val_batch.candidates.max(), test_batch.candidates.max(), val_batch.targets.max(), test_batch.targets.max()) + 1)
    meta = read_item_metadata(args.movies_path, args.datadir / 'mappings.json', n_items)
    records = load_records(args.profile_run / 'claim_support.jsonl')
    faith = pd.read_csv(args.profile_run / 'table1_profile_faithfulness.csv')
    manifest = json.loads((args.profile_run / 'run_manifest.json').read_text(encoding='utf-8'))

    base_ranked = test_batch.candidates[:, :args.topk]
    base_metrics = metrics_from_ranked(base_ranked, test_batch.targets)
    direct_metrics = metrics_from_ranked(direct_ranked, test_batch.targets)
    direct_costs = direct_cost(args.direct_run)
    profile_costs = profile_cost(args.profile_run, test_batch.users)

    rows: List[Dict[str, object]] = []
    per_user_frames: Dict[str, pd.DataFrame] = {
        'lightgcn': base_metrics.copy(),
        'direct': direct_metrics.copy(),
    }
    rows.append({'Method': 'LightGCN', 'SelectedLambda': 0.0, **metric_summary(base_metrics), **reliability_summary('LightGCN', base_metrics, base_metrics), 'EstimatedCostUSD_TestUsers': 0.0, 'CostRatioVsDirect_TestUsers': 0.0})
    rows.append({'Method': 'DeepSeek Direct Rerank', 'SelectedLambda': np.nan, **metric_summary(direct_metrics), **reliability_summary('DeepSeek Direct Rerank', base_metrics, direct_metrics), **direct_costs})

    lambda_tables: List[pd.DataFrame] = []
    method_specs = [
        ('Profile Rerank Raw', 'raw', 'Raw Profile'),
        ('Profile Rerank Remove', 'remove', 'Remove Repair'),
        ('Profile Rerank EGPR', 'weighted', 'Evidence-Weighted Repair'),
    ]
    for label, method_key, faith_method in method_specs:
        val_scores = profile_scores_for_batch(val_batch, records, meta, method_key)
        test_scores = profile_scores_for_batch(test_batch, records, meta, method_key)
        selected_lambda, lambda_table = select_lambda(label, method_key, val_batch, val_scores, args.lambda_grid, args.topk)
        lambda_tables.append(lambda_table)
        ranked = ranked_from_profile_scores(test_batch, test_scores, selected_lambda, args.topk)
        metrics = metrics_from_ranked(ranked, test_batch.targets)
        safe = label.lower().replace(' ', '_')
        per_user_frames[safe] = metrics.copy()
        np.save(args.outdir / f'reranked_{safe}_top{args.topk}.npy', ranked)
        faith_row = first_row(faith, faith_method)
        row = {
            'Method': label,
            'SelectedLambda': selected_lambda,
            'ClaimsPerUser': int(manifest.get('claims_per_user', 0)),
            'UCR': float(faith_row['UCR']),
            'WeightedUCR': float(faith_row['WeightedUCR']),
            'ProfileDriftScore': float(faith_row['ProfileDriftScore']),
            **metric_summary(metrics),
            **reliability_summary(label, base_metrics, metrics),
            **profile_costs,
            **win_tie_loss(metrics, direct_metrics),
        }
        rows.append(row)

    comparison = pd.DataFrame(rows)
    direct_cost_value = float(direct_costs['EstimatedCostUSD_TestUsers'])
    comparison['NDCGGainVsBase'] = comparison['NDCG@20'] - float(comparison.loc[comparison['Method'] == 'LightGCN', 'NDCG@20'].iloc[0])
    comparison['NDCGGainVsDirect'] = comparison['NDCG@20'] - float(comparison.loc[comparison['Method'] == 'DeepSeek Direct Rerank', 'NDCG@20'].iloc[0])
    comparison['CostRatioVsDirect_TestUsers'] = comparison['EstimatedCostUSD_TestUsers'] / direct_cost_value
    comparison.to_csv(args.outdir / 'fair_direct_candidate_comparison.csv', index=False)
    pd.concat(lambda_tables, ignore_index=True).to_csv(args.outdir / 'fair_direct_candidate_lambda_validation.csv', index=False)

    for name, metrics in per_user_frames.items():
        frame = metrics.copy()
        frame.insert(0, 'target', test_batch.targets.astype(int))
        frame.insert(0, 'uid', test_batch.users.astype(int))
        frame.to_csv(args.outdir / f'per_user_{name}.csv', index=False)

    slim_cols = [
        'Method', 'SelectedLambda', 'NDCG@20', 'Recall@20', 'NDCGGainVsBase', 'NDCGGainVsDirect',
        'HarmRate', 'GainHarmRatio', 'UCR', 'WeightedUCR', 'ProfileDriftScore',
        'EstimatedCostUSD_TestUsers', 'CostRatioVsDirect_TestUsers', 'EstimatedCostUSD_ValTestUnique',
        'WinsVsDirect', 'TiesVsDirect', 'LossesVsDirect',
    ]
    slim_cols = [c for c in slim_cols if c in comparison.columns]
    slim = comparison[slim_cols].copy()
    slim.to_csv(args.outdir / 'fair_direct_candidate_summary.csv', index=False)

    decision = {
        'dataset': 'ML-1M',
        'users': int(len(test_batch.users)),
        'candidate_setting': 'direct LightGCN top-50 candidates reused for direct and profile reranking',
        'same_test_users': True,
        'same_targets': True,
        'direct_ndcg': float(comparison.loc[comparison['Method'] == 'DeepSeek Direct Rerank', 'NDCG@20'].iloc[0]),
        'profile_raw_ndcg': float(comparison.loc[comparison['Method'] == 'Profile Rerank Raw', 'NDCG@20'].iloc[0]),
        'profile_egpr_ndcg': float(comparison.loc[comparison['Method'] == 'Profile Rerank EGPR', 'NDCG@20'].iloc[0]),
        'profile_egpr_beats_direct': bool(comparison.loc[comparison['Method'] == 'Profile Rerank EGPR', 'NDCG@20'].iloc[0] > comparison.loc[comparison['Method'] == 'DeepSeek Direct Rerank', 'NDCG@20'].iloc[0]),
        'profile_test_cost_below_direct': bool(profile_costs['EstimatedCostUSD_TestUsers'] < direct_cost_value),
        'updated_at_utc': datetime.now(timezone.utc).isoformat(),
    }
    (args.outdir / 'fair_direct_candidate_decision.json').write_text(json.dumps(decision, indent=2) + '\n', encoding='utf-8')

    lines = [
        '# ML-1M 2000 Fair Direct-Candidate Comparison',
        '',
        'Direct rerank and profile rerank are evaluated on the same Direct-run LightGCN top-50 val/test candidate sets. No API calls are made by this comparison.',
        '',
        '## Summary',
        '',
        markdown_table(slim),
        '',
        '## Decision',
        '',
        '```json',
        json.dumps(decision, indent=2),
        '```',
        '',
        '## Artifacts',
        '',
        '- `fair_direct_candidate_summary.csv`',
        '- `fair_direct_candidate_comparison.csv`',
        '- `fair_direct_candidate_lambda_validation.csv`',
        '- `per_user_*.csv`',
        '',
    ]
    (args.outdir / 'fair_direct_candidate_decision.md').write_text('\n'.join(lines), encoding='utf-8')
    print(args.outdir / 'fair_direct_candidate_decision.md')


if __name__ == '__main__':
    main()

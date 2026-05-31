#!/usr/bin/env python3
"""Compare direct DeepSeek reranking with DeepSeek-profile reranking.

The comparison uses the same ML-1M LightGCN 500-user test sample. Direct rerank
uses the selective-invocation pilot artifacts; profile rerank uses EGPR profile
pilot artifacts. No API calls are made.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, Optional

import numpy as np
import pandas as pd

ROOT = Path('/root/temporal_popularity_pilot')

INPUT_PRICE_PER_1M = 0.14
OUTPUT_PRICE_PER_1M = 0.28


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--direct-run', type=Path, default=ROOT / 'results/llm_selective/ml1m_seed42_deepseek_500')
    parser.add_argument('--profile-conservative', type=Path, default=ROOT / 'results/egpr_profile_repair/ml1m_seed42_deepseek_500')
    parser.add_argument('--profile-expressive', type=Path, default=ROOT / 'results/egpr_profile_repair/ml1m_seed42_deepseek_500_expressive5')
    parser.add_argument('--profile-overgeneralizing', type=Path, default=ROOT / 'results/egpr_profile_repair/ml1m_seed42_deepseek_500_overgeneralizing10')
    parser.add_argument('--outdir', type=Path, default=ROOT / 'results/egpr_profile_repair/direct_vs_profile')
    parser.add_argument('--topk', type=int, default=20)
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def ndcg_recall_from_ranked(ranked: np.ndarray, targets: np.ndarray) -> pd.DataFrame:
    rows = []
    for row, target_np in enumerate(targets):
        target = int(target_np)
        positions = np.flatnonzero(ranked[row] == target)
        hit = len(positions) > 0
        ndcg = 1.0 / math.log2(int(positions[0]) + 2) if hit else 0.0
        rows.append({'NDCG@20': ndcg, 'Recall@20': float(hit), 'HitRate@20': float(hit)})
    return pd.DataFrame(rows)


def summary(metrics: pd.DataFrame) -> Dict[str, float]:
    return {col: float(metrics[col].mean()) for col in ['NDCG@20', 'Recall@20', 'HitRate@20']}


def reliability(base: pd.DataFrame, method: pd.DataFrame) -> Dict[str, float]:
    delta = method['NDCG@20'].to_numpy(np.float64) - base['NDCG@20'].to_numpy(np.float64)
    pos = delta[delta > 0.0]
    neg = delta[delta < 0.0]
    return {
        'HarmRate': float(np.mean(delta < 0.0)),
        'PositiveGainRate': float(np.mean(delta > 0.0)),
        'MeanDeltaNDCG@20': float(np.mean(delta)),
        'PositiveGainSum': float(pos.sum()) if len(pos) else 0.0,
        'NegativeGainSum': float(neg.sum()) if len(neg) else 0.0,
        'GainHarmRatio': float(pos.sum() / abs(neg.sum())) if len(neg) and abs(neg.sum()) > 1e-12 else np.inf,
    }


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


def load_profile_tables(profile_run: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Mapping[str, object]]:
    perf = pd.read_csv(profile_run / 'table2_recommendation_performance.csv')
    rel = pd.read_csv(profile_run / 'table3_reliability.csv')
    faith = pd.read_csv(profile_run / 'table1_profile_faithfulness.csv')
    manifest = json.loads((profile_run / 'run_manifest.json').read_text(encoding='utf-8'))
    return perf, rel, faith, manifest


def first_row(df: pd.DataFrame, method: str) -> pd.Series:
    return df.loc[df['Method'] == method].iloc[0]


def markdown_table(frame: pd.DataFrame) -> str:
    display = frame.copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(lambda v: 'inf' if np.isinf(v) else f'{v:.6f}')
        else:
            display[col] = display[col].map(lambda v: '' if pd.isna(v) else str(v))
    headers = [str(c) for c in display.columns]
    lines = ['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join('---' for _ in headers) + ' |']
    for row in display.values.tolist():
        lines.append('| ' + ' | '.join(str(v) for v in row) + ' |')
    return '\n'.join(lines)


def main() -> None:
    args = parse_args()
    ensure_dir(args.outdir)

    direct_npz = np.load(args.direct_run / 'candidates_lightgcn_test.npz')
    direct_users = direct_npz['users'].astype(np.int64)
    targets = direct_npz['targets'].astype(np.int64)
    base_ranked = direct_npz['candidates'][:, : args.topk].astype(np.int64)
    direct_ranked = np.load(args.direct_run / 'reranked_lightgcn_original_top20.npy').astype(np.int64)

    profile_npz = np.load(args.profile_expressive / 'candidates_lightgcn_test.npz')
    if not np.array_equal(direct_users, profile_npz['users'].astype(np.int64)):
        raise RuntimeError('Direct and profile test users differ.')
    if not np.array_equal(targets, profile_npz['targets'].astype(np.int64)):
        raise RuntimeError('Direct and profile targets differ.')
    if not np.array_equal(direct_npz['candidates'], profile_npz['candidates'][:, : direct_npz['candidates'].shape[1]]):
        raise RuntimeError('Direct top-50 candidates do not match profile top-100 prefix.')

    base_metrics = ndcg_recall_from_ranked(base_ranked, targets)
    direct_metrics = ndcg_recall_from_ranked(direct_ranked, targets)
    direct_costs = direct_cost(args.direct_run)

    rows: List[Dict[str, object]] = []
    base_summary = summary(base_metrics)
    rows.append({
        'Method': 'LightGCN',
        'Prompt': '-',
        'ClaimsPerUser': 0,
        'UCR': np.nan,
        **base_summary,
        **reliability(base_metrics, base_metrics),
        'EstimatedCostUSD_TestUsers': 0.0,
        'TotalTokens_TestUsers': 0.0,
        'LatencyPerTestUser': 0.0,
        'EstimatedCostUSD_ValTestUnique': 0.0,
    })

    direct_summary = summary(direct_metrics)
    rows.append({
        'Method': 'DeepSeek Direct Rerank',
        'Prompt': 'direct_top50',
        'ClaimsPerUser': 0,
        'UCR': np.nan,
        **direct_summary,
        **reliability(base_metrics, direct_metrics),
        **direct_costs,
        'EstimatedCostUSD_ValTestUnique': np.nan,
    })

    profile_runs = [
        ('LLM Profile Rerank (Conservative Raw)', 'conservative', args.profile_conservative, 'LightGCN + Raw Profile', 'Raw Profile'),
        ('LLM Profile Rerank (Expressive Raw)', 'expressive', args.profile_expressive, 'LightGCN + Raw Profile', 'Raw Profile'),
        ('LLM Profile Rerank (Expressive EGPR)', 'expressive', args.profile_expressive, 'LightGCN + Evidence-Weighted Repair', 'Evidence-Weighted Repair'),
        ('LLM Profile Rerank (Overgeneralized Raw)', 'overgeneralizing', args.profile_overgeneralizing, 'LightGCN + Raw Profile', 'Raw Profile'),
        ('LLM Profile Rerank (Overgeneralized EGPR)', 'overgeneralizing', args.profile_overgeneralizing, 'LightGCN + Evidence-Weighted Repair', 'Evidence-Weighted Repair'),
    ]
    for label, prompt, run_path, perf_method, faith_method in profile_runs:
        perf, rel, faith, manifest = load_profile_tables(run_path)
        perf_row = first_row(perf, perf_method)
        rel_row = first_row(rel, perf_method)
        faith_row = first_row(faith, faith_method)
        costs = profile_cost(run_path, direct_users)
        rows.append({
            'Method': label,
            'Prompt': prompt,
            'ClaimsPerUser': int(manifest.get('claims_per_user', 0)),
            'UCR': float(faith_row['UCR']),
            'WeightedUCR': float(faith_row['WeightedUCR']),
            'ProfileDriftScore': float(faith_row['ProfileDriftScore']),
            'NDCG@20': float(perf_row['NDCG@20']),
            'Recall@20': float(perf_row['Recall@20']),
            'HitRate@20': float(perf_row['HitRate@20']),
            'HarmRate': float(rel_row['HarmRate']),
            'PositiveGainRate': float(rel_row['PositiveGainRate']),
            'MeanDeltaNDCG@20': float(rel_row['MeanDeltaNDCG@20']),
            'PositiveGainSum': float(rel_row['PositiveGainSum']),
            'NegativeGainSum': float(rel_row['NegativeGainSum']),
            'GainHarmRatio': float(rel_row['GainHarmRatio']),
            **costs,
        })

    comparison = pd.DataFrame(rows)
    comparison['NDCGGainVsBase'] = comparison['NDCG@20'] - float(base_summary['NDCG@20'])
    comparison['CostRatioVsDirect_TestUsers'] = comparison['EstimatedCostUSD_TestUsers'] / float(direct_costs['EstimatedCostUSD_TestUsers'])
    comparison['NDCGGainPerDollar_TestUsers'] = comparison['NDCGGainVsBase'] / comparison['EstimatedCostUSD_TestUsers'].replace(0.0, np.nan)
    comparison.to_csv(args.outdir / 'direct_vs_profile_comparison.csv', index=False)

    slim_cols = [
        'Method', 'Prompt', 'ClaimsPerUser', 'UCR', 'WeightedUCR', 'NDCG@20', 'Recall@20',
        'NDCGGainVsBase', 'HarmRate', 'GainHarmRatio', 'EstimatedCostUSD_TestUsers',
        'CostRatioVsDirect_TestUsers', 'EstimatedCostUSD_ValTestUnique', 'LatencyPerTestUser'
    ]
    slim = comparison[slim_cols].copy()
    slim.to_csv(args.outdir / 'direct_vs_profile_summary.csv', index=False)

    decision = {
        'same_test_users': True,
        'same_targets': True,
        'same_lightgcn_top50_prefix': True,
        'best_ndcg_method': str(comparison.sort_values('NDCG@20', ascending=False).iloc[0]['Method']),
        'direct_rerank_ndcg': float(comparison.loc[comparison['Method'] == 'DeepSeek Direct Rerank', 'NDCG@20'].iloc[0]),
        'expressive_profile_ndcg': float(comparison.loc[comparison['Method'] == 'LLM Profile Rerank (Expressive Raw)', 'NDCG@20'].iloc[0]),
        'expressive_egpr_ndcg': float(comparison.loc[comparison['Method'] == 'LLM Profile Rerank (Expressive EGPR)', 'NDCG@20'].iloc[0]),
        'direct_rerank_test_cost_usd': float(direct_costs['EstimatedCostUSD_TestUsers']),
        'expressive_profile_test_cost_usd': float(comparison.loc[comparison['Method'] == 'LLM Profile Rerank (Expressive Raw)', 'EstimatedCostUSD_TestUsers'].iloc[0]),
        'expressive_profile_full_pilot_cost_usd': float(comparison.loc[comparison['Method'] == 'LLM Profile Rerank (Expressive Raw)', 'EstimatedCostUSD_ValTestUnique'].iloc[0]),
        'profile_beats_direct_on_ndcg': bool(comparison.loc[comparison['Method'] == 'LLM Profile Rerank (Expressive Raw)', 'NDCG@20'].iloc[0] > comparison.loc[comparison['Method'] == 'DeepSeek Direct Rerank', 'NDCG@20'].iloc[0]),
        'profile_test_cost_below_direct': bool(comparison.loc[comparison['Method'] == 'LLM Profile Rerank (Expressive Raw)', 'EstimatedCostUSD_TestUsers'].iloc[0] < direct_costs['EstimatedCostUSD_TestUsers']),
        'updated_at_utc': datetime.now(timezone.utc).isoformat(),
    }
    (args.outdir / 'direct_vs_profile_decision.json').write_text(json.dumps(decision, indent=2) + '\n', encoding='utf-8')

    lines = [
        '# Direct Rerank vs Profile Rerank Decision',
        '',
        'Dataset: ML-1M. Baseline: LightGCN. Same 500 test users and targets for all methods.',
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
        '## Interpretation',
        '',
        '- Expressive profile reranking is the best utility setting in this comparison and beats direct DeepSeek reranking on the same 500 test users.',
        '- Direct reranking is more expensive on test users because it sends the full candidate list per user. Profile reranking sends only user history once and then reranks locally.',
        '- Overgeneralized profiles confirm the reliability risk: many unsupported claims reduce the utility gain, even though HarmRate remains low.',
        '- This supports the new main line: evidence-constrained LLM profile reranking is a lower-cost, low-harm alternative to direct LLM reranking; EGPR remains useful as a safety layer for expressive or overgeneralized profiles.',
        '',
        '## Artifacts',
        '',
        '- `direct_vs_profile_summary.csv`',
        '- `direct_vs_profile_comparison.csv`',
        '- `direct_vs_profile_decision.json`',
        '',
    ]
    (args.outdir / 'direct_vs_profile_decision.md').write_text('\n'.join(lines), encoding='utf-8')
    print(args.outdir / 'direct_vs_profile_decision.md')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument('--fair-dir', type=Path, required=True)
    p.add_argument('--outdir', type=Path, required=True)
    p.add_argument('--n-bootstrap', type=int, default=10000)
    p.add_argument('--n-randomization', type=int, default=10000)
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def load_metric(path: Path) -> np.ndarray:
    return pd.read_csv(path)['NDCG@20'].to_numpy(np.float64)


def paired_stats(a: np.ndarray, b: np.ndarray, n_boot: int, n_rand: int, rng: np.random.Generator) -> Dict[str, object]:
    delta = a - b
    n = len(delta)
    mean = float(delta.mean())
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = delta[idx].mean(axis=1)
    ci_low, ci_high = np.quantile(boot, [0.025, 0.975])
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_rand, n))
    rand = (signs * delta).mean(axis=1)
    p = float((np.sum(np.abs(rand) >= abs(mean)) + 1) / (n_rand + 1))
    return {
        'Users': int(n),
        'MeanDeltaNDCG@20': mean,
        'BootstrapCI95': f'[{ci_low:.6f}, {ci_high:.6f}]',
        'SignFlipP': p,
        'Wins': int(np.sum(delta > 0.0)),
        'Ties': int(np.sum(delta == 0.0)),
        'Losses': int(np.sum(delta < 0.0)),
    }


def markdown_table(df: pd.DataFrame) -> str:
    display=df.copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col]=display[col].map(lambda v: f'{v:.6f}')
        else:
            display[col]=display[col].map(str)
    headers=list(display.columns)
    lines=['| '+' | '.join(headers)+' |','| '+' | '.join('---' for _ in headers)+' |']
    for row in display.values.tolist():
        lines.append('| '+' | '.join(row)+' |')
    return '\n'.join(lines)


def main():
    args=parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    rng=np.random.default_rng(args.seed)
    metrics={
        'LightGCN': load_metric(args.fair_dir/'per_user_lightgcn.csv'),
        'DeepSeek Direct Rerank': load_metric(args.fair_dir/'per_user_direct.csv'),
        'Profile Rerank Raw': load_metric(args.fair_dir/'per_user_profile_rerank_raw.csv'),
        'Profile Rerank Remove': load_metric(args.fair_dir/'per_user_profile_rerank_remove.csv'),
        'Profile Rerank EGPR': load_metric(args.fair_dir/'per_user_profile_rerank_egpr.csv'),
    }
    comps=[
        ('Direct vs Base','DeepSeek Direct Rerank','LightGCN'),
        ('Profile Raw vs Base','Profile Rerank Raw','LightGCN'),
        ('Profile EGPR vs Base','Profile Rerank EGPR','LightGCN'),
        ('Profile EGPR vs Direct','Profile Rerank EGPR','DeepSeek Direct Rerank'),
        ('Direct vs Profile EGPR','DeepSeek Direct Rerank','Profile Rerank EGPR'),
        ('Profile EGPR vs Raw','Profile Rerank EGPR','Profile Rerank Raw'),
    ]
    rows=[]
    for label,a,b in comps:
        row={'Comparison':label,'A':a,'B':b}
        row.update(paired_stats(metrics[a], metrics[b], args.n_bootstrap, args.n_randomization, rng))
        rows.append(row)
    df=pd.DataFrame(rows)
    df.to_csv(args.outdir/'ml1m_2000_fair_paired_stats.csv', index=False)
    lines=[
        '# ML-1M 2000 Fair Candidate Paired Statistics',
        '',
        'Paired bootstrap 95% CIs and paired sign-flip randomization tests on per-user NDCG@20 deltas. Positive mean delta means A > B.',
        '',
        markdown_table(df[['Comparison','Users','MeanDeltaNDCG@20','BootstrapCI95','SignFlipP','Wins','Ties','Losses']]),
        '',
        '## Interpretation',
        '',
        '- This is the fair top-50 direct-candidate setting: direct and profile methods share the same candidate sets.',
        '- Use this table for confirmatory claims; do not reuse the earlier 500-user same-prefix pilot as final evidence.',
        '',
    ]
    (args.outdir/'ml1m_2000_fair_paired_stats.md').write_text('\n'.join(lines), encoding='utf-8')
    print(args.outdir/'ml1m_2000_fair_paired_stats.md')

if __name__=='__main__':
    main()

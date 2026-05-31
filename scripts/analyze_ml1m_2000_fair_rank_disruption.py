#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument('--direct-run', type=Path, required=True)
    p.add_argument('--fair-dir', type=Path, required=True)
    p.add_argument('--outdir', type=Path, required=True)
    p.add_argument('--topk', type=int, default=20)
    return p.parse_args()


def metrics(ranked: np.ndarray, targets: np.ndarray) -> pd.DataFrame:
    rows=[]
    for row,t in enumerate(targets.astype(int)):
        pos=np.flatnonzero(ranked[row]==t)
        hit=len(pos)>0
        ndcg=1/np.log2(int(pos[0])+2) if hit else 0.0
        rows.append({'NDCG@20':float(ndcg),'Recall@20':float(hit)})
    return pd.DataFrame(rows)


def rank_disruption(base: np.ndarray, method: np.ndarray) -> Dict[str,float]:
    overlaps=[]; jaccards=[]; shifts=[]; changed=[]
    for b,m in zip(base,method):
        bs=set(map(int,b)); ms=set(map(int,m))
        inter=len(bs & ms)
        union=len(bs | ms)
        overlaps.append(inter/len(b))
        jaccards.append(inter/union if union else 1.0)
        changed.append(float(not np.array_equal(b,m)))
        bpos={int(i):p for p,i in enumerate(b)}
        local=[]
        for p,i in enumerate(m):
            ii=int(i)
            if ii in bpos:
                local.append(abs(p-bpos[ii]))
            else:
                local.append(len(b))
        shifts.append(float(np.mean(local)))
    return {
        'ChangedRankingRateVsBase':float(np.mean(changed)),
        'Top20OverlapVsBase':float(np.mean(overlaps)),
        'Top20JaccardVsBase':float(np.mean(jaccards)),
        'MeanAbsRankShiftVsBase':float(np.mean(shifts)),
    }


def summary(label, ranked, base_ranked, targets, base_metrics):
    met=metrics(ranked,targets)
    delta=met['NDCG@20'].to_numpy()-base_metrics['NDCG@20'].to_numpy()
    row={'Method':label,'Users':len(targets),'NDCG@20':float(met['NDCG@20'].mean()),'Recall@20':float(met['Recall@20'].mean()),
         'MeanDeltaVsBase':float(delta.mean()),'WinsVsBase':int((delta>0).sum()),'TiesVsBase':int((delta==0).sum()),'LossesVsBase':int((delta<0).sum()),
         'HitUsers':int((met['Recall@20']>0).sum())}
    row.update(rank_disruption(base_ranked,ranked))
    return row, met


def markdown_table(df):
    d=df.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c]=d[c].map(lambda v:f'{v:.6f}')
        else:
            d[c]=d[c].map(str)
    lines=['| '+' | '.join(d.columns)+' |','| '+' | '.join('---' for _ in d.columns)+' |']
    for row in d.values.tolist():
        lines.append('| '+' | '.join(row)+' |')
    return '\n'.join(lines)


def main():
    args=parse_args(); args.outdir.mkdir(parents=True,exist_ok=True)
    cand=np.load(args.direct_run/'candidates_lightgcn_test.npz')
    targets=cand['targets'].astype(int)
    base=cand['candidates'][:,:args.topk].astype(int)
    methods={
        'LightGCN':base,
        'DeepSeek Direct Rerank':np.load(args.direct_run/f'reranked_lightgcn_original_top{args.topk}.npy').astype(int),
        'Profile Rerank Raw':np.load(args.fair_dir/f'reranked_profile_rerank_raw_top{args.topk}.npy').astype(int),
        'Profile Rerank Remove':np.load(args.fair_dir/f'reranked_profile_rerank_remove_top{args.topk}.npy').astype(int),
        'Profile Rerank EGPR':np.load(args.fair_dir/f'reranked_profile_rerank_egpr_top{args.topk}.npy').astype(int),
    }
    base_metrics=metrics(base,targets)
    rows=[]; per={}
    for label,ranked in methods.items():
        row,met=summary(label,ranked,base,targets,base_metrics)
        rows.append(row); per[label]=met
    df=pd.DataFrame(rows)
    df.to_csv(args.outdir/'ml1m_2000_fair_rank_disruption.csv',index=False)
    # affected users: any hit under any method
    hit_mask=np.zeros(len(targets),dtype=bool)
    delta_mask=np.zeros(len(targets),dtype=bool)
    for label,met in per.items():
        hit_mask |= met['Recall@20'].to_numpy()>0
        delta_mask |= met['NDCG@20'].to_numpy()!=base_metrics['NDCG@20'].to_numpy()
    subset_rows=[]
    for subset_name,mask in [('any_method_hit',hit_mask),('any_ndcg_delta_vs_base',delta_mask)]:
        for label,met in per.items():
            sub=met.loc[mask]
            bsub=base_metrics.loc[mask]
            delta=sub['NDCG@20'].to_numpy()-bsub['NDCG@20'].to_numpy()
            subset_rows.append({'Subset':subset_name,'Method':label,'Users':int(mask.sum()),'NDCG@20':float(sub['NDCG@20'].mean()),'Recall@20':float(sub['Recall@20'].mean()),'MeanDeltaVsBase':float(delta.mean()),'WinsVsBase':int((delta>0).sum()),'TiesVsBase':int((delta==0).sum()),'LossesVsBase':int((delta<0).sum())})
    sdf=pd.DataFrame(subset_rows)
    sdf.to_csv(args.outdir/'ml1m_2000_fair_affected_subsets.csv',index=False)
    lines=['# ML-1M 2000 Fair Rank-Disruption Analysis','','## Method-Level', '', markdown_table(df), '', '## Affected Subsets', '', markdown_table(sdf), '']
    (args.outdir/'ml1m_2000_fair_rank_disruption.md').write_text('\n'.join(lines),encoding='utf-8')
    print(args.outdir/'ml1m_2000_fair_rank_disruption.md')

if __name__=='__main__':
    main()

# Yelp Profile Rerank Mini-Pilot

Profile mode: `api`. Provider: `deepseek`.
Dataset: Yelp. Baseline: LightGCN seed42. Candidate set: top-100. Output: top-20.
Users per split: 1000. History limit: 20. Claims per user: 5. Prompt variant: expressive.
Estimated profile generation cost USD: 0.286402.

## Profile Faithfulness

| Method | Claims | UCR | WeightedUCR | EvidenceCoverage | ProfileDriftScore |
| --- | --- | --- | --- | --- | --- |
| Raw Profile | 9965 | 0.089413 | 0.089413 | 0.832615 | 0.657084 |
| Remove Repair | 9074 | 0.000000 | 0.000000 | 0.832615 | 0.653332 |
| Evidence-Weighted Repair | 9965 | 0.089413 | 0.011045 | 0.832615 | 0.643282 |

## Recommendation Performance

| Method | SelectedLambda | NDCG@20 | Recall@20 | HitRate@20 |
| --- | --- | --- | --- | --- |
| LightGCN | 0.000000 | 0.015520 | 0.037000 | 0.037000 |
| LightGCN + Raw Profile | 1.000000 | 0.016560 | 0.036000 | 0.036000 |
| LightGCN + Remove Repair | 1.000000 | 0.016006 | 0.036000 | 0.036000 |
| LightGCN + Evidence-Weighted Repair | 1.000000 | 0.015245 | 0.033000 | 0.033000 |

## Reliability

| Method | HarmRate | PositiveGainRate | MeanDeltaNDCG@20 | PositiveGainSum | NegativeGainSum | GainHarmRatio |
| --- | --- | --- | --- | --- | --- | --- |
| LightGCN | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | nan |
| LightGCN + Raw Profile | 0.019000 | 0.021000 | 0.001040 | 5.508580 | -4.468768 | 1.232684 |
| LightGCN + Remove Repair | 0.019000 | 0.021000 | 0.000486 | 4.911061 | -4.424944 | 1.109858 |
| LightGCN + Evidence-Weighted Repair | 0.020000 | 0.021000 | -0.000275 | 4.936654 | -5.211484 | 0.947264 |

## Decision Signals

```json
{
  "profile_rerank_beats_base": true,
  "egpr_ndcg_ge_or_close_to_raw": false,
  "raw_ucr_ge_10pct": false,
  "egpr_weighted_ucr_relative_drop_ge_30pct": true,
  "egpr_harm_below_or_close_to_raw": true,
  "egpr_ghr_above_raw": false,
  "pass_count": 3,
  "profile_mode": "api",
  "base_ndcg": 0.015520207835529926,
  "raw_ndcg": 0.01656001993663025,
  "egpr_ndcg": 0.015245377250613111,
  "raw_ucr": 0.08941294530858003,
  "egpr_weighted_ucr": 0.011044725933980688,
  "raw_harm": 0.019,
  "egpr_harm": 0.02,
  "raw_ghr": 1.2326842844757027,
  "egpr_ghr": 0.9472644322235718
}
```

## Artifacts

- `user_history.jsonl`
- `raw_profiles.jsonl`
- `claim_support.jsonl`
- `repaired_profiles.jsonl`
- `table1_profile_faithfulness.csv`
- `table2_recommendation_performance.csv`
- `table3_reliability.csv`
- `table4_lambda_validation.csv`
- `profile_cost_trace.csv`
- `run_manifest.json`

# EGPR ML-1M Mini-Pilot

Profile mode: `api`.
Dataset: ML-1M. Baseline: LightGCN. Candidate set: top-100. Output: top-20.
Users per split: 2000. History limit: 20. Claims per user: 5. Prompt variant: expressive.

## Profile Faithfulness

| Method | Claims | UCR | WeightedUCR | EvidenceCoverage | ProfileDriftScore |
| --- | --- | --- | --- | --- | --- |
| Raw Profile | 16760 | 0.122673 | 0.122673 | 0.979390 | 0.240059 |
| Remove Repair | 14704 | 0.000000 | 0.000000 | 0.979390 | 0.226881 |
| Evidence-Weighted Repair | 16760 | 0.122673 | 0.008692 | 0.979390 | 0.225747 |

## Recommendation Performance

| Method | SelectedLambda | NDCG@20 | Recall@20 | HitRate@20 |
| --- | --- | --- | --- | --- |
| LightGCN | 0.000000 | 0.024856 | 0.065500 | 0.065500 |
| LightGCN + Raw Profile | 1.000000 | 0.029233 | 0.074000 | 0.074000 |
| LightGCN + Remove Repair | 1.000000 | 0.029346 | 0.074500 | 0.074500 |
| LightGCN + Evidence-Weighted Repair | 1.000000 | 0.029108 | 0.073500 | 0.073500 |

## Reliability

| Method | HarmRate | PositiveGainRate | MeanDeltaNDCG@20 | PositiveGainSum | NegativeGainSum | GainHarmRatio |
| --- | --- | --- | --- | --- | --- | --- |
| LightGCN | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | nan |
| LightGCN + Raw Profile | 0.036500 | 0.050500 | 0.004377 | 24.619036 | -15.865801 | 1.551705 |
| LightGCN + Remove Repair | 0.036500 | 0.051000 | 0.004490 | 24.846706 | -15.865801 | 1.566054 |
| LightGCN + Evidence-Weighted Repair | 0.036500 | 0.050000 | 0.004252 | 24.395499 | -15.891335 | 1.535145 |

## Go / No-Go

```json
{
  "raw_ucr_ge_15pct": false,
  "remove_ucr_relative_drop_ge_30pct": true,
  "egpr_weighted_ucr_relative_drop_ge_30pct": true,
  "raw_harm_ge_25pct": false,
  "egpr_harm_below_raw": false,
  "egpr_ndcg_ge_or_close_to_raw": true,
  "egpr_ghr_above_raw": false,
  "critical_signal_present": false,
  "pass_count": 3,
  "decision": "no_go_or_revise",
  "raw_ucr": 0.12267303102625299,
  "remove_ucr": 0.0,
  "egpr_weighted_ucr": 0.00869150261552175,
  "raw_harm": 0.0365,
  "egpr_harm": 0.0365,
  "raw_ndcg": 0.029232527565667167,
  "egpr_ndcg": 0.02910799253559647,
  "raw_ghr": 1.5517045193305028,
  "egpr_ghr": 1.5351447402651377
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
- `go_no_go.json`
- `run_manifest.json`

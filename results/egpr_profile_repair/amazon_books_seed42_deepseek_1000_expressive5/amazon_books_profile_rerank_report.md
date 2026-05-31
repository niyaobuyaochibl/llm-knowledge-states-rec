# Amazon Books Profile Rerank Pilot

Dataset: Amazon Books subset. Baseline: LightGCN. Candidate set: top-100. Output: top-20.
Users per split: 1000. History limit: 20. Claims per user: 5.
Estimated profile generation cost USD: 0.168051.

## Profile Faithfulness

| Method | Claims | UCR | WeightedUCR | EvidenceCoverage | ProfileDriftScore |
| --- | --- | --- | --- | --- | --- |
| Raw Profile | 8230 | 0.347995 | 0.347995 | 0.469567 | 0.836464 |
| Remove Repair | 5366 | 0.000000 | 0.000000 | 0.469567 | 0.826732 |
| Evidence-Weighted Repair | 8230 | 0.347995 | 0.127873 | 0.469567 | 0.817824 |

## Recommendation Performance

| Method | SelectedLambda | NDCG@20 | Recall@20 | HitRate@20 |
| --- | --- | --- | --- | --- |
| LightGCN | 0.000000 | 0.004263 | 0.011000 | 0.011000 |
| LightGCN + Raw Profile | 0.500000 | 0.004864 | 0.012000 | 0.012000 |
| LightGCN + Remove Repair | 0.300000 | 0.004975 | 0.013000 | 0.013000 |
| LightGCN + Evidence-Weighted Repair | 0.500000 | 0.004812 | 0.011000 | 0.011000 |

## Reliability

| Method | HarmRate | PositiveGainRate | MeanDeltaNDCG@20 | PositiveGainSum | NegativeGainSum | GainHarmRatio |
| --- | --- | --- | --- | --- | --- | --- |
| LightGCN | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | nan |
| LightGCN + Raw Profile | 0.005000 | 0.005000 | 0.000600 | 0.920757 | -0.320571 | 2.872237 |
| LightGCN + Remove Repair | 0.002000 | 0.004000 | 0.000712 | 0.785721 | -0.074069 | 10.608022 |
| LightGCN + Evidence-Weighted Repair | 0.005000 | 0.006000 | 0.000549 | 1.119611 | -0.570571 | 1.962263 |

## Artifacts

- `raw_profiles.jsonl`
- `claim_support.jsonl`
- `table1_profile_faithfulness.csv`
- `table2_recommendation_performance.csv`
- `table3_reliability.csv`
- `table4_lambda_validation.csv`
- `profile_cost_trace.csv`
- `run_manifest.json`

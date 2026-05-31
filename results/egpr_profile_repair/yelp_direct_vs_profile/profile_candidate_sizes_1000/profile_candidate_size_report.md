# Yelp Profile Rerank Candidate-Size Evaluation

Profile run: `/root/temporal_popularity_pilot/results/egpr_profile_repair/yelp_seed42_deepseek_1000_expressive5`.
Candidate sizes: [50, 100]. Output top-20.

## Performance

| CandidateSet | Method | SelectedLambda | NDCG@20 | Recall@20 | HitRate@20 |
| --- | --- | --- | --- | --- | --- |
| 50 | LightGCN | 0.000000 | 0.015520 | 0.037000 | 0.037000 |
| 50 | LightGCN + Raw Profile | 1.000000 | 0.016009 | 0.035000 | 0.035000 |
| 50 | LightGCN + Remove Repair | 1.000000 | 0.015398 | 0.034000 | 0.034000 |
| 50 | LightGCN + Evidence-Weighted Repair | 1.000000 | 0.015258 | 0.034000 | 0.034000 |
| 100 | LightGCN | 0.000000 | 0.015520 | 0.037000 | 0.037000 |
| 100 | LightGCN + Raw Profile | 1.000000 | 0.016560 | 0.036000 | 0.036000 |
| 100 | LightGCN + Remove Repair | 1.000000 | 0.016006 | 0.036000 | 0.036000 |
| 100 | LightGCN + Evidence-Weighted Repair | 1.000000 | 0.015245 | 0.033000 | 0.033000 |

## Reliability

| CandidateSet | Method | HarmRate | PositiveGainRate | MeanDeltaNDCG@20 | PositiveGainSum | NegativeGainSum | GainHarmRatio |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 50 | LightGCN | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | - |
| 50 | LightGCN + Raw Profile | 0.020000 | 0.020000 | 0.000489 | 4.835130 | -4.346178 | 1.112502 |
| 50 | LightGCN + Remove Repair | 0.019000 | 0.020000 | -0.000122 | 4.381243 | -4.503087 | 0.972942 |
| 50 | LightGCN + Evidence-Weighted Repair | 0.019000 | 0.020000 | -0.000263 | 4.138050 | -4.400733 | 0.940309 |
| 100 | LightGCN | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | - |
| 100 | LightGCN + Raw Profile | 0.019000 | 0.021000 | 0.001040 | 5.508580 | -4.468768 | 1.232684 |
| 100 | LightGCN + Remove Repair | 0.019000 | 0.021000 | 0.000486 | 4.911061 | -4.424944 | 1.109858 |
| 100 | LightGCN + Evidence-Weighted Repair | 0.020000 | 0.021000 | -0.000275 | 4.936654 | -5.211484 | 0.947264 |

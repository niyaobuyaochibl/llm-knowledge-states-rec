# Yelp 1000 Fair Confirmatory Analysis

Profile run: `/root/temporal_popularity_pilot/results/egpr_profile_repair/yelp_seed42_deepseek_1000_expressive5`.
Direct rerank and profile rerank are evaluated on matched LightGCN top-50 and top-100 candidate sets.
Profile generation cost is reusable and fixed across candidate sizes: `0.286402` USD.

## Fair Accuracy / Reliability / Cost

| CandidateSet | Method | NDCG@20 | Recall@20 | DeltaVsBase | HarmRate | GHR | TestCostUSD | CostVsDirect |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 50 | LightGCN | 0.015520 | 0.037000 | 0.000000 | 0.000000 | - | 0.000000 | 0.000000 |
| 50 | DeepSeek Direct Rerank | 0.014380 | 0.034000 | -0.001140 | 0.016000 | 0.807183 | 0.344530 | 1.000000 |
| 50 | Profile Rerank Raw | 0.016009 | 0.035000 | 0.000489 | 0.020000 | 1.112502 | 0.286402 | 0.831285 |
| 50 | Profile Rerank Remove | 0.015398 | 0.034000 | -0.000122 | 0.019000 | 0.972942 | 0.286402 | 0.831285 |
| 50 | Profile Rerank EGPR | 0.015258 | 0.034000 | -0.000263 | 0.019000 | 0.940309 | 0.286402 | 0.831285 |
| 100 | LightGCN | 0.015520 | 0.037000 | 0.000000 | 0.000000 | - | 0.000000 | 0.000000 |
| 100 | DeepSeek Direct Rerank | 0.015210 | 0.036000 | -0.000310 | 0.016000 | 0.942263 | 0.610509 | 1.000000 |
| 100 | Profile Rerank Raw | 0.016560 | 0.036000 | 0.001040 | 0.019000 | 1.232684 | 0.286402 | 0.469120 |
| 100 | Profile Rerank Remove | 0.016006 | 0.036000 | 0.000486 | 0.019000 | 1.109858 | 0.286402 | 0.469120 |
| 100 | Profile Rerank EGPR | 0.015245 | 0.033000 | -0.000275 | 0.020000 | 0.947264 | 0.286402 | 0.469120 |

## Paired NDCG Tests

| CandidateSet | Comparison | Metric | Users | MeanDelta | BootstrapCI95Low | BootstrapCI95High | SignFlipPValueTwoSided | Wins | Ties | Losses |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 50 | Direct vs Base | NDCG@20 | 1000 | -0.001140 | -0.005660 | 0.003000 | 0.615884 | 22 | 962 | 16 |
| 50 | Profile Raw vs Base | NDCG@20 | 1000 | 0.000489 | -0.002941 | 0.004045 | 0.784162 | 20 | 960 | 20 |
| 50 | Profile EGPR vs Base | NDCG@20 | 1000 | -0.000263 | -0.003475 | 0.002895 | 0.872561 | 20 | 961 | 19 |
| 50 | Profile Raw vs Direct | NDCG@20 | 1000 | 0.001629 | -0.002856 | 0.006167 | 0.482475 | 21 | 958 | 21 |
| 50 | Profile EGPR vs Direct | NDCG@20 | 1000 | 0.000878 | -0.003258 | 0.005097 | 0.690483 | 20 | 958 | 22 |
| 50 | Profile EGPR vs Raw | NDCG@20 | 1000 | -0.000752 | -0.002283 | 0.000447 | 0.346397 | 8 | 980 | 12 |
| 100 | Direct vs Base | NDCG@20 | 1000 | -0.000310 | -0.004700 | 0.003827 | 0.888361 | 22 | 962 | 16 |
| 100 | Profile Raw vs Base | NDCG@20 | 1000 | 0.001040 | -0.002620 | 0.005004 | 0.596124 | 21 | 960 | 19 |
| 100 | Profile EGPR vs Base | NDCG@20 | 1000 | -0.000275 | -0.003735 | 0.003333 | 0.882121 | 21 | 959 | 20 |
| 100 | Profile Raw vs Direct | NDCG@20 | 1000 | 0.001350 | -0.003472 | 0.006362 | 0.593474 | 21 | 954 | 25 |
| 100 | Profile EGPR vs Direct | NDCG@20 | 1000 | 0.000036 | -0.004425 | 0.004745 | 0.988430 | 19 | 956 | 25 |
| 100 | Profile EGPR vs Raw | NDCG@20 | 1000 | -0.001315 | -0.003128 | 0.000092 | 0.108099 | 10 | 978 | 12 |

## Rank Disruption

| CandidateSet | Method | Users | NDCG@20 | Recall@20 | WinsVsBase | TiesVsBase | LossesVsBase | Top20OverlapVsBase | Top20JaccardVsBase | MeanAbsRankShiftVsBase | HitUsers |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 50 | LightGCN | 1000 | 0.015520 | 0.037000 | 0 | 1000 | 0 | 1.000000 | 1.000000 | 0.000000 | 37 |
| 50 | DeepSeek Direct Rerank | 1000 | 0.014380 | 0.034000 | 22 | 962 | 16 | 0.703250 | 0.587177 | 4.200314 | 34 |
| 50 | Profile Rerank Raw | 1000 | 0.016009 | 0.035000 | 20 | 960 | 20 | 0.688200 | 0.529289 | 4.290632 | 35 |
| 50 | Profile Rerank Remove | 1000 | 0.015398 | 0.034000 | 20 | 961 | 19 | 0.686750 | 0.527727 | 4.304368 | 34 |
| 50 | Profile Rerank EGPR | 1000 | 0.015258 | 0.034000 | 20 | 961 | 19 | 0.684900 | 0.525681 | 4.321275 | 34 |
| 100 | LightGCN | 1000 | 0.015520 | 0.037000 | 0 | 1000 | 0 | 1.000000 | 1.000000 | 0.000000 | 37 |
| 100 | DeepSeek Direct Rerank | 1000 | 0.015210 | 0.036000 | 22 | 962 | 16 | 0.605300 | 0.510200 | 4.195716 | 36 |
| 100 | Profile Rerank Raw | 1000 | 0.016560 | 0.036000 | 21 | 960 | 19 | 0.627700 | 0.462468 | 4.326051 | 36 |
| 100 | Profile Rerank Remove | 1000 | 0.016006 | 0.036000 | 21 | 960 | 19 | 0.624900 | 0.459613 | 4.347545 | 36 |
| 100 | Profile Rerank EGPR | 1000 | 0.015245 | 0.033000 | 21 | 959 | 20 | 0.622950 | 0.457517 | 4.384661 | 33 |

## Affected-Hit Subset

| CandidateSet | Subset | Method | Users | NDCG@20 | Recall@20 | MeanDeltaVsBase | WinsVsBase | TiesVsBase | LossesVsBase |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 50 | any_method_hit | LightGCN | 49 | 0.316739 | 0.755102 | 0.000000 | 0 | 49 | 0 |
| 50 | any_method_hit | DeepSeek Direct Rerank | 49 | 0.293465 | 0.693878 | -0.023274 | 22 | 11 | 16 |
| 50 | any_method_hit | Profile Rerank Raw | 49 | 0.326718 | 0.714286 | 0.009979 | 20 | 9 | 20 |
| 50 | any_method_hit | Profile Rerank Remove | 49 | 0.314252 | 0.693878 | -0.002487 | 20 | 10 | 19 |
| 50 | any_method_hit | Profile Rerank EGPR | 49 | 0.311378 | 0.693878 | -0.005361 | 20 | 10 | 19 |
| 100 | any_method_hit | LightGCN | 52 | 0.298466 | 0.711538 | 0.000000 | 0 | 52 | 0 |
| 100 | any_method_hit | DeepSeek Direct Rerank | 52 | 0.292497 | 0.692308 | -0.005968 | 22 | 14 | 16 |
| 100 | any_method_hit | Profile Rerank Raw | 52 | 0.318462 | 0.692308 | 0.019996 | 21 | 12 | 19 |
| 100 | any_method_hit | Profile Rerank Remove | 52 | 0.307814 | 0.692308 | 0.009348 | 21 | 12 | 19 |
| 100 | any_method_hit | Profile Rerank EGPR | 52 | 0.293180 | 0.634615 | -0.005285 | 21 | 11 | 20 |

## Interpretation

- On 1000 Yelp users, Direct DeepSeek reranking remains below LightGCN under both matched top-50 and matched top-100 candidate sets.
- Raw expressive profile reranking is the strongest profile setting on Yelp 1000; Evidence-Weighted EGPR reduces unsupported-claim weight but harms utility relative to Raw.
- Profile reranking remains cheaper than Direct, especially under top-100 candidates, but its rank-disruption advantage is not uniform on Yelp 1000; the main stable advantage is lower cost with better Raw-profile utility than Direct.

## Artifacts

- `yelp_1000_fair_summary.csv`
- `yelp_1000_paired_stats.csv`
- `yelp_1000_rank_disruption.csv`
- `yelp_1000_affected_subset.csv`
- `run_manifest.json`

# ML-1M 2000 Fair Rank-Disruption Analysis

## Method-Level

| Method | Users | NDCG@20 | Recall@20 | MeanDeltaVsBase | WinsVsBase | TiesVsBase | LossesVsBase | HitUsers | ChangedRankingRateVsBase | Top20OverlapVsBase | Top20JaccardVsBase | MeanAbsRankShiftVsBase |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LightGCN | 2000 | 0.024856 | 0.065500 | 0.000000 | 0 | 2000 | 0 | 131 | 0.000000 | 1.000000 | 1.000000 | 0.000000 |
| DeepSeek Direct Rerank | 2000 | 0.031196 | 0.074500 | 0.006340 | 118 | 1810 | 72 | 149 | 0.990000 | 0.515200 | 0.358128 | 12.902000 |
| Profile Rerank Raw | 2000 | 0.027805 | 0.069500 | 0.002950 | 89 | 1841 | 70 | 139 | 1.000000 | 0.699425 | 0.542785 | 9.461450 |
| Profile Rerank Remove | 2000 | 0.027805 | 0.069500 | 0.002950 | 89 | 1841 | 70 | 139 | 1.000000 | 0.699450 | 0.542855 | 9.464825 |
| Profile Rerank EGPR | 2000 | 0.027985 | 0.069500 | 0.003129 | 88 | 1841 | 71 | 139 | 1.000000 | 0.698750 | 0.542146 | 9.478800 |

## Affected Subsets

| Subset | Method | Users | NDCG@20 | Recall@20 | MeanDeltaVsBase | WinsVsBase | TiesVsBase | LossesVsBase |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| any_method_hit | LightGCN | 208 | 0.238999 | 0.629808 | 0.000000 | 0 | 208 | 0 |
| any_method_hit | DeepSeek Direct Rerank | 208 | 0.299962 | 0.716346 | 0.060963 | 118 | 18 | 72 |
| any_method_hit | Profile Rerank Raw | 208 | 0.267360 | 0.668269 | 0.028361 | 89 | 49 | 70 |
| any_method_hit | Profile Rerank Remove | 208 | 0.267360 | 0.668269 | 0.028361 | 89 | 49 | 70 |
| any_method_hit | Profile Rerank EGPR | 208 | 0.269086 | 0.668269 | 0.030087 | 88 | 49 | 71 |
| any_ndcg_delta_vs_base | LightGCN | 205 | 0.227863 | 0.624390 | 0.000000 | 0 | 205 | 0 |
| any_ndcg_delta_vs_base | DeepSeek Direct Rerank | 205 | 0.289718 | 0.712195 | 0.061855 | 118 | 15 | 72 |
| any_ndcg_delta_vs_base | Profile Rerank Raw | 205 | 0.256639 | 0.663415 | 0.028776 | 89 | 46 | 70 |
| any_ndcg_delta_vs_base | Profile Rerank Remove | 205 | 0.256639 | 0.663415 | 0.028776 | 89 | 46 | 70 |
| any_ndcg_delta_vs_base | Profile Rerank EGPR | 205 | 0.258390 | 0.663415 | 0.030527 | 88 | 46 | 71 |

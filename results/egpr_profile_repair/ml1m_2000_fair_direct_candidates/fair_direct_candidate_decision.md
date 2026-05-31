# ML-1M 2000 Fair Direct-Candidate Comparison

Direct rerank and profile rerank are evaluated on the same Direct-run LightGCN top-50 val/test candidate sets. No API calls are made by this comparison.

## Summary

| Method | SelectedLambda | NDCG@20 | Recall@20 | NDCGGainVsBase | NDCGGainVsDirect | HarmRate | GainHarmRatio | UCR | WeightedUCR | ProfileDriftScore | EstimatedCostUSD_TestUsers | CostRatioVsDirect_TestUsers | EstimatedCostUSD_ValTestUnique | WinsVsDirect | TiesVsDirect | LossesVsDirect |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LightGCN | 0.000000 | 0.024856 | 0.065500 | 0.000000 | -0.006340 | 0.000000 | inf |  |  |  | 0.000000 | 0.000000 |  |  |  |  |
| DeepSeek Direct Rerank |  | 0.031196 | 0.074500 | 0.006340 | 0.000000 | 0.036000 | 1.568273 |  |  |  | 0.596293 | 1.000000 |  |  |  |  |
| Profile Rerank Raw | 1.000000 | 0.027805 | 0.069500 | 0.002950 | -0.003391 | 0.035000 | 1.420942 | 0.122673 | 0.122673 | 0.240059 | 0.266001 | 0.446091 | 0.445671 | 69.000000 | 1822.000000 | 109.000000 |
| Profile Rerank Remove | 1.000000 | 0.027805 | 0.069500 | 0.002950 | -0.003391 | 0.035000 | 1.420942 | 0.000000 | 0.000000 | 0.226881 | 0.266001 | 0.446091 | 0.445671 | 69.000000 | 1822.000000 | 109.000000 |
| Profile Rerank EGPR | 1.000000 | 0.027985 | 0.069500 | 0.003129 | -0.003211 | 0.035500 | 1.444589 | 0.122673 | 0.008692 | 0.225747 | 0.266001 | 0.446091 | 0.445671 | 70.000000 | 1820.000000 | 110.000000 |

## Decision

```json
{
  "dataset": "ML-1M",
  "users": 2000,
  "candidate_setting": "direct LightGCN top-50 candidates reused for direct and profile reranking",
  "same_test_users": true,
  "same_targets": true,
  "direct_ndcg": 0.03119607385806312,
  "profile_raw_ndcg": 0.02780546785623266,
  "profile_egpr_ndcg": 0.02798493013171209,
  "profile_egpr_beats_direct": false,
  "profile_test_cost_below_direct": true,
  "updated_at_utc": "2026-05-20T06:38:51.340177+00:00"
}
```

## Artifacts

- `fair_direct_candidate_summary.csv`
- `fair_direct_candidate_comparison.csv`
- `fair_direct_candidate_lambda_validation.csv`
- `per_user_*.csv`

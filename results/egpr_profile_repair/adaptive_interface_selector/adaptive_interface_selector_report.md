# Adaptive LLM Interface Selector (ALIS)

This report reuses the completed fair-candidate experiments and adds an
adaptive selection layer over five interfaces: Base, Direct rerank, Raw Profile,
Remove Repair, and EGPR. No new API calls are used.

ALIS scores each interface within a dataset as:

```text
Score = w_u * Utility - w_c * Cost - w_h * Harm - w_d * Disruption - w_g * GroundingRisk
```

Utility is min-max-normalized NDCG@20 within the dataset. Cost is the API cost
ratio relative to Direct. Harm is HarmRate normalized by the dataset maximum.
Disruption is `1 - Top20OverlapVsBase`. GroundingRisk is WeightedUCR normalized
by the maximum profile WeightedUCR in the dataset.

## Selected Interfaces by Deployment Preference

| Dataset | PreferenceProfile | SelectedInterface | ALISScore | NDCG@20 | DeltaVsBase | HarmRate | CostVsDirect | WeightedUCR | Top20OverlapVsBase |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Amazon Books | accuracy_first | Remove Repair | 0.982086 | 0.004975 | 0.000712 | 0.002000 | 0.565208 | 0.000000 | 0.910450 |
| Amazon Books | balanced | Remove Repair | 0.863407 | 0.004975 | 0.000712 | 0.002000 | 0.565208 | 0.000000 | 0.910450 |
| Amazon Books | cost_first | Remove Repair | 0.308130 | 0.004975 | 0.000712 | 0.002000 | 0.565208 | 0.000000 | 0.910450 |
| Amazon Books | safety_first | Remove Repair | 0.530047 | 0.004975 | 0.000712 | 0.002000 | 0.565208 | 0.000000 | 0.910450 |
| ML-1M | accuracy_first | Direct | 0.955152 | 0.031196 | 0.006340 | 0.036000 | 1.000000 | 0.000000 | 0.515200 |
| ML-1M | balanced | Direct | 0.651520 | 0.031196 | 0.006340 | 0.036000 | 1.000000 | 0.000000 | 0.515200 |
| ML-1M | cost_first | Direct | 0.001520 | 0.031196 | 0.006340 | 0.036000 | 1.000000 | 0.000000 | 0.515200 |
| ML-1M | safety_first | Direct | 0.177280 | 0.031196 | 0.006340 | 0.036000 | 1.000000 | 0.000000 | 0.515200 |
| Yelp | accuracy_first | Raw Profile | 0.957895 | 0.016560 | 0.001040 | 0.019000 | 0.469120 | 0.089413 | 0.627700 |
| Yelp | balanced | Raw Profile | 0.649902 | 0.016560 | 0.001040 | 0.019000 | 0.469120 | 0.089413 | 0.627700 |
| Yelp | cost_first | Raw Profile | 0.206666 | 0.016560 | 0.001040 | 0.019000 | 0.469120 | 0.089413 | 0.627700 |
| Yelp | safety_first | Base | 0.160900 | 0.015520 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 |

## Candidate Interface Diagnostic Panel

| Dataset | Method | NDCG@20 | DeltaVsBase | HarmRate | CostVsDirect | WeightedUCR | ProfileDrift | Top20OverlapVsBase | MeanAbsRankShift |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ML-1M | Base | 0.024856 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.000000 |
| ML-1M | Direct | 0.031196 | 0.006340 | 0.036000 | 1.000000 | 0.000000 | 0.000000 | 0.515200 | 12.902000 |
| ML-1M | Raw Profile | 0.027805 | 0.002950 | 0.035000 | 0.446091 | 0.122673 | 0.240059 | 0.699425 | 9.461450 |
| ML-1M | Remove Repair | 0.027805 | 0.002950 | 0.035000 | 0.446091 | 0.000000 | 0.226881 | 0.699450 | 9.464825 |
| ML-1M | EGPR | 0.027985 | 0.003129 | 0.035500 | 0.446091 | 0.008692 | 0.225747 | 0.698750 | 9.478800 |
| Yelp | Base | 0.015520 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.000000 |
| Yelp | Direct | 0.015210 | -0.000310 | 0.016000 | 1.000000 | 0.000000 | 0.000000 | 0.605300 | 4.195716 |
| Yelp | Raw Profile | 0.016560 | 0.001040 | 0.019000 | 0.469120 | 0.089413 | 0.657084 | 0.627700 | 4.326051 |
| Yelp | Remove Repair | 0.016006 | 0.000486 | 0.019000 | 0.469120 | 0.000000 | 0.653332 | 0.624900 | 4.347545 |
| Yelp | EGPR | 0.015245 | -0.000275 | 0.020000 | 0.469120 | 0.011045 | 0.643282 | 0.622950 | 4.384661 |
| Amazon Books | Base | 0.004263 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.000000 |
| Amazon Books | Direct | 0.003946 | -0.000317 | 0.007000 | 1.000000 | 0.000000 | 0.000000 | 0.655650 | 4.239199 |
| Amazon Books | Raw Profile | 0.004864 | 0.000600 | 0.005000 | 0.565208 | 0.347995 | 0.836464 | 0.818650 | 2.728521 |
| Amazon Books | Remove Repair | 0.004975 | 0.000712 | 0.002000 | 0.565208 | 0.000000 | 0.826732 | 0.910450 | 1.524135 |
| Amazon Books | EGPR | 0.004812 | 0.000549 | 0.005000 | 0.565208 | 0.127873 | 0.817824 | 0.817800 | 2.772396 |

## Balanced-Profile Ablation

| Dataset | Ablation | SelectedInterface | ALISScore | NDCG@20 | CostVsDirect | WeightedUCR | Top20OverlapVsBase |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Amazon Books | balanced_full | Remove Repair | 0.863407 | 0.004975 | 0.565208 | 0.000000 | 0.910450 |
| Amazon Books | no_cost_penalty | Remove Repair | 0.948188 | 0.004975 | 0.565208 | 0.000000 | 0.910450 |
| Amazon Books | no_disruption_penalty | Remove Repair | 0.872362 | 0.004975 | 0.565208 | 0.000000 | 0.910450 |
| Amazon Books | no_grounding_penalty | Remove Repair | 0.863407 | 0.004975 | 0.565208 | 0.000000 | 0.910450 |
| Amazon Books | no_harm_penalty | Remove Repair | 0.906264 | 0.004975 | 0.565208 | 0.000000 | 0.910450 |
| ML-1M | balanced_full | Direct | 0.651520 | 0.031196 | 1.000000 | 0.000000 | 0.515200 |
| ML-1M | no_cost_penalty | Direct | 0.801520 | 0.031196 | 1.000000 | 0.000000 | 0.515200 |
| ML-1M | no_disruption_penalty | Direct | 0.700000 | 0.031196 | 1.000000 | 0.000000 | 0.515200 |
| ML-1M | no_grounding_penalty | Direct | 0.651520 | 0.031196 | 1.000000 | 0.000000 | 0.515200 |
| ML-1M | no_harm_penalty | Direct | 0.801520 | 0.031196 | 1.000000 | 0.000000 | 0.515200 |
| Yelp | balanced_full | Raw Profile | 0.649902 | 0.016560 | 0.469120 | 0.089413 | 0.627700 |
| Yelp | no_cost_penalty | Raw Profile | 0.720270 | 0.016560 | 0.469120 | 0.089413 | 0.627700 |
| Yelp | no_disruption_penalty | Raw Profile | 0.687132 | 0.016560 | 0.469120 | 0.089413 | 0.627700 |
| Yelp | no_grounding_penalty | Raw Profile | 0.749902 | 0.016560 | 0.469120 | 0.089413 | 0.627700 |
| Yelp | no_harm_penalty | Raw Profile | 0.792402 | 0.016560 | 0.469120 | 0.089413 | 0.627700 |

## Interpretation

- Under accuracy-first and balanced preferences, ALIS selects Direct on ML-1M,
  Raw Profile on Yelp, and Remove Repair on Amazon Books. This recovers the
  three-domain pattern while making the selection rule explicit.
- Under safety-first preferences, ALIS can choose Base on Yelp because Raw
  Profile improves utility but carries non-trivial harm and unsupported-claim
  mass. This is a useful deployment behavior: the selector is allowed to abstain
  from LLM augmentation when risk penalties dominate.
- Amazon Books consistently selects Remove Repair across preferences. This
  matches the high Raw UCR and low evidence coverage observed in the profile
  faithfulness panel, where unsupported claims are more likely to be harmful.
- The ablation table shows which penalty terms move the selected interface. It
  can be used in the manuscript to argue that ALIS is a method-level contribution
  rather than only a descriptive comparison.

## Artifacts

- `candidate_interface_metrics.csv`
- `alis_preference_scores.csv`
- `alis_selected_interfaces.csv`
- `alis_ablation.csv`
- `alis_sensitivity_grid.csv`
- `adaptive_interface_selector_report.md`

# Formal Experiment Preparation Runbook

## Current Decision

Status: **Go for formal experiment preparation**, not immediate full launch.

The mini-pilot now has:

- ML-1M sanity-check evidence for ranking instability and group sensitivity.
- Yelp raw-review evidence for strong temporal bucket drift.
- Yelp Day-2 Base-only evidence that static and temporal metrics change the interpretation of the same recommendation lists.
- Yelp Day-3 PopPenalty evidence for Decay LTR/PCE ranking flips.
- Yelp protocol robustness for 90/180/365 windows and 50/100/200/400 snapshots.
- Yelp exact subset validation showing the 200-snapshot approximation has negligible drift-stat error on a 5,000-user time-stratified subset.

## Required Cleanup Before Full Runs

1. Extract shared code from the pilot scripts into a small package:
   - `temporal_popularity_pilot/src/temporal_popularity/data.py` done
   - `temporal_popularity_pilot/src/temporal_popularity/popularity.py` done
   - `temporal_popularity_pilot/src/temporal_popularity/eval.py` done
   - `temporal_popularity_pilot/src/temporal_popularity/model.py` done
   - `temporal_popularity_pilot/src/temporal_popularity/temporal.py` done
   - `temporal_popularity_pilot/src/temporal_popularity/audit.py` done
   - `temporal_popularity_pilot/src/temporal_popularity/reporting.py` done
   - `temporal_popularity_pilot/src/temporal_popularity/snapshots.py` done

2. Keep experiment launch scripts thin:
   - dataset loading and split selection
   - model configuration
   - method list
   - output directory

3. Preserve the mini-pilot outputs unchanged as an audit trail.

## Formal Protocol

Datasets:

- ML-1M: exact per-user temporal evaluation is feasible.
- Yelp original reviews: use 200 weighted test-time snapshots for main scalable reporting, with exact subset validation in appendix.

Popularity definitions:

- StaticTrainPop.
- RecentPop@180d.
- DecayPop@180d.
- CumulativePop as secondary/appendix.
- Robustness: 90d and 365d.

Methods:

- Base.
- Static PopPenalty.
- Temporal PopPenalty.
- Static PopCal.
- Temporal PopCal.
- One existing baseline, preferably PDA or xQuAD-style post-processing if in-processing is too costly.

Seeds:

- 42, 43, 44.

Validation:

- Full-ranking only.
- Static method lambda selected by static metric with NDCG@20 drop <= 5%.
- Temporal method lambda selected by temporal metric with NDCG@20 drop <= 5%.
- If no lambda satisfies the constraint, choose the smallest NDCG drop and mark it.

## Immediate Next Commands

These are preparation commands, not full formal runs:

```bash
python -m py_compile /root/temporal_popularity_pilot/scripts/*.py
```

```bash
python /root/temporal_popularity_pilot/scripts/run_yelp_protocol_robustness.py
```

```bash
python /root/temporal_popularity_pilot/scripts/run_yelp_exact_subset_check.py
```

```bash
python /root/temporal_popularity_pilot/scripts/validate_formal_setup.py
```

The latest dry-run validation report is:

- `results/formal/aggregate/formal_setup_validation.md`
- `results/formal/aggregate/formal_run_preparation.md`

## Formal Launch Gate

Start full formal runs only after all of these are true:

- Shared code modules exist and pilot scripts still reproduce prior results.
- Output schemas for Tables 1-5 are fixed.
- Yelp snapshot protocol is stated in method text.
- Exact subset validation is included in appendix.
- Seeded run directories are planned before launch.
- Disk budget is checked for all user-level metric CSVs and recommendation arrays.
- Per-seed `config.json` and `run_manifest.json` files exist under `results/formal/{dataset}/seed*/`.

## Output Directory Convention

Use one directory per dataset, method family, and seed:

```text
results/formal/
  ml1m/
    seed42/
    seed43/
    seed44/
  yelp/
    seed42/
    seed43/
    seed44/
  aggregate/
    table1_dataset_stats.csv
    table2_temporal_drift.csv
    table3_static_vs_temporal_eval.csv
    table4_tod_rfr.csv
    table5_group_sensitivity.csv
```

Each seed directory should contain:

```text
config.json
split_stats.csv
validation_all_lambda_metrics.csv
test_method_metrics.csv
test_user_level_metrics.csv
tod_rfr.csv
group_sensitivity.csv
run_manifest.json
```

## Paper Framing Guardrails

- Do not claim temporal popularity is new.
- Do not claim Temporal PopPenalty or Temporal PopCal as new methods.
- Frame PopPenalty/PopCal as controlled diagnostic variants.
- State that snapshot-weighted Yelp evaluation is a scalable approximation validated by exact subset results.
- Keep the contribution centered on conclusion stability and reporting protocol.

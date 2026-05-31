#!/usr/bin/env bash
set -euo pipefail

: "${DEEPSEEK_API_KEY:?Set DEEPSEEK_API_KEY in the environment before running this script.}"

ROOT=/root/temporal_popularity_pilot
PROFILE_RUN="$ROOT/results/egpr_profile_repair/ml1m_seed42_deepseek_2000_expressive5"
DIRECT_RUN="$ROOT/results/llm_selective/ml1m_seed42_deepseek_2000_h20"
COMPARE_RUN="$ROOT/results/egpr_profile_repair/ml1m_direct_vs_profile_2000_h20"
PROFILE_CACHE="$ROOT/results/egpr_profile_repair/ml1m_seed42_deepseek_500_expressive5/profile_api_cache"
DIRECT_CACHE="$ROOT/results/llm_selective/ml1m_seed42_deepseek_2000_h20/api_cache"

python "$ROOT/scripts/run_egpr_profile_repair_pilot.py" \
  --profile-mode api \
  --provider deepseek \
  --prompt-variant expressive \
  --claims-per-user 5 \
  --max-users 2000 \
  --top-candidates 100 \
  --history-limit 20 \
  --outdir "$PROFILE_RUN" \
  --cache-dir "$PROFILE_CACHE"

python "$ROOT/scripts/run_llm_selective_invocation_pilot.py" \
  --mode api \
  --provider deepseek \
  --baselines lightgcn \
  --max-users 2000 \
  --top-candidates 50 \
  --topk 20 \
  --history-limit 20 \
  --outdir "$DIRECT_RUN" \
  --figdir "$ROOT/figures/llm_selective/ml1m_seed42_deepseek_2000_h20" \
  --cache-dir "$DIRECT_CACHE"

python "$ROOT/scripts/compare_ml1m_profile_before_ranking.py" \
  --direct-run "$DIRECT_RUN" \
  --profile-run "$PROFILE_RUN" \
  --outdir "$COMPARE_RUN" \
  --profile-label Expressive

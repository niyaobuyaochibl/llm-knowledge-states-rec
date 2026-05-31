# ALIS: Adaptive Knowledge-Interface Selection for LLM-Assisted Recommendation

Companion code and redistributable artefacts for the research project:

> **ALIS: Adaptive Knowledge-Interface Selection for LLM-Assisted Recommender Systems.**  
> Yunan Zhang, Jingjing Fan, Yanxiao Liu.

This repository is journal-neutral. It is intended as a replication package for
studying how large language models (LLMs) should be connected to recommender
systems through different *knowledge interfaces*.

The project reformulates LLM-assisted recommendation as an adaptive
knowledge-interface selection problem. ALIS compares several states an LLM can
expose to a recommender system:

- **Base**: the backbone recommender without LLM augmentation;
- **Direct**: a transient candidate-conditioned LLM reranking interface;
- **Raw Profile**: a reusable LLM-generated user-profile knowledge artefact;
- **Remove Repair**: hard removal of weakly supported profile claims;
- **EGPR**: evidence-weighted profile repair.

The evaluation covers ML-1M, Yelp, and Amazon Books under matched candidate
sets, and reports diagnostics spanning ranking utility, downside risk, API cost,
rank disruption, and profile faithfulness. A SASRec check evaluates whether the
cached profile knowledge artefact transfers beyond the LightGCN backbone.

## Repository layout

```
configs/                Experiment configuration templates.
src/temporal_popularity/
                        Core library: data loading, ranking metrics,
                        LightGCN / SASRec backbones, popularity audits,
                        temporal snapshots, and reporting utilities.
scripts/                Experiment runners and analyzers, including backbone
                        training, direct LLM reranking, profile generation,
                        evidence-guided repair, paired statistics, candidate
                        matching, and ALIS selection.
manuscript_figures/     Stand-alone Python scripts that regenerate paper
                        figures from the artefacts in results/.
results/                Aggregated, redistributable result artefacts used in
                        the study: per-method NDCG/Recall, validation tables,
                        faithfulness diagnostics, claim-support logs, paired
                        statistics, rank-disruption traces, cost traces, and
                        ALIS selection tables. Per-user CSV files use anonymous
                        user indices only; raw original-dataset rows are not
                        redistributed.
docs_FORMAL_EXPERIMENT_PREP.md
                        Experiment preparation notes retained for transparency.
```

## What is not redistributed

To respect dataset licensing and user-data governance, the following are **not**
redistributed and must be obtained from their original sources:

- **Raw datasets.** ML-1M is available from GroupLens; the Amazon review corpora
  from Julian McAuley's site; and Yelp from the Yelp Open Dataset. Original
  interaction logs are not stored in this repository.
- **Trained checkpoints.** LightGCN and SASRec checkpoints can be reproduced
  from `scripts/` using the configs in `configs/`.
- **Private API credentials.** LLM calls require your own provider key. Do not
  commit API keys or SSH credentials to this repository.

## Reproducing the study

1. Install dependencies with Python 3.10+:

   ```bash
   pip install numpy pandas scipy scikit-learn torch sentence-transformers \
               matplotlib pyyaml requests tqdm
   ```

2. Obtain the benchmark datasets and place them under `data/` following the
   layout expected by `src/temporal_popularity/data.py`.

3. Train backbones and produce candidate sets:

   ```bash
   bash scripts/run_ml1m_confirmatory_2000.sh
   python scripts/train_amazon_books_lightgcn_candidates.py
   ```

4. Generate profiles, run direct reranking, and apply evidence-guided repair
   using the `scripts/run_*_profile_rerank_pilot.py` and
   `scripts/run_*_direct_rerank_pilot.py` scripts. DeepSeek API access is
   required for the LLM steps; set the key in the environment as
   `DEEPSEEK_API_KEY` before running.

5. Aggregate and analyze paired comparisons:

   ```bash
   python scripts/bootstrap_ml1m_2000_fair.py
   python scripts/analyze_yelp_1000_fair_confirmatory.py
   python scripts/analyze_amazon_books_fair_confirmatory.py
   ```

6. Build the ALIS adaptive knowledge-interface selection panel:

   ```bash
   python scripts/run_adaptive_interface_selector.py
   ```

   This step does not call any LLM API. It reuses the completed matched
   evaluation artefacts and writes ALIS scores, selected interfaces, ablations,
   and sensitivity grids under
   `results/egpr_profile_repair/adaptive_interface_selector/`.

7. Regenerate figures from the artefacts in `results/`:

   ```bash
   python manuscript_figures/generate_figures.py
   python manuscript_figures/generate_sensitivity_figures.py
   ```

## Main reproducibility artefacts

The most relevant ALIS outputs are:

- `results/egpr_profile_repair/adaptive_interface_selector/candidate_interface_metrics.csv`
- `results/egpr_profile_repair/adaptive_interface_selector/alis_preference_scores.csv`
- `results/egpr_profile_repair/adaptive_interface_selector/alis_selected_interfaces.csv`
- `results/egpr_profile_repair/adaptive_interface_selector/alis_ablation.csv`
- `results/egpr_profile_repair/adaptive_interface_selector/alis_sensitivity_grid.csv`
- `results/egpr_profile_repair/adaptive_interface_selector/adaptive_interface_selector_report.md`

## License

Code is released under the MIT License (see `LICENSE`). The aggregated result
artefacts under `results/` are released under
[Creative Commons Attribution 4.0 (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)
and contain only model-derived statistics; no original dataset rows are
redistributed.

## Contact

Corresponding author: **Jingjing Fan** (`fjj1960@hebiace.edu.cn`), Hebei
University of Architecture, Zhangjiakou, Hebei, China.

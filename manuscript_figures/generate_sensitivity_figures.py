"""
Generates λ-sensitivity and τ-sensitivity figures from the real experiment
artifacts under ../../../results/egpr_profile_repair/.

Outputs:
  figures/lambda_sensitivity.pdf
  figures/tau_sensitivity.pdf
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
PROJECT_ROOT = HERE.parents[2]
RESULTS_ROOT = PROJECT_ROOT / "results" / "egpr_profile_repair"

mpl.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
})

METHOD_COLOR = {
    "Profile-Raw": "#1f77b4",
    "Profile-Remove": "#2ca02c",
    "Profile-EGPR": "#9467bd",
}
METHOD_MARKER = {
    "Profile-Raw": "o",
    "Profile-Remove": "D",
    "Profile-EGPR": "^",
}
DATASET_BASE = {
    "ML-1M (top-50)": 0.024856,
    "Yelp (top-100)": 0.015520,
    "Amazon Books (top-100)": 0.004263,
}


# -------------------------------------------------------------------------
# Lambda sensitivity
# -------------------------------------------------------------------------

def _load_ml1m_lambda():
    fp = RESULTS_ROOT / "ml1m_2000_fair_direct_candidates" / "fair_direct_candidate_lambda_validation.csv"
    rows = _read_csv(fp)
    keymap = {"raw": "Profile-Raw", "remove": "Profile-Remove", "weighted": "Profile-EGPR"}
    out = {v: {"lambda": [], "ndcg": []} for v in keymap.values()}
    for r in rows:
        m = keymap.get(r["method_key"])
        if m is None:
            continue
        out[m]["lambda"].append(float(r["lambda"]))
        out[m]["ndcg"].append(float(r["NDCG@20"]))
    return out


def _load_yelp_lambda(cand_set="100"):
    fp = RESULTS_ROOT / "yelp_direct_vs_profile" / "profile_candidate_sizes_1000" / "profile_candidate_lambda_validation.csv"
    rows = _read_csv(fp)
    keymap = {"raw": "Profile-Raw", "remove": "Profile-Remove", "weighted": "Profile-EGPR"}
    out = {v: {"lambda": [], "ndcg": []} for v in keymap.values()}
    for r in rows:
        if r["CandidateSet"] != cand_set:
            continue
        m = keymap.get(r["method_key"])
        if m is None:
            continue
        out[m]["lambda"].append(float(r["lambda"]))
        out[m]["ndcg"].append(float(r["NDCG@20"]))
    return out


def _load_amazon_lambda():
    fp = RESULTS_ROOT / "amazon_books_seed42_deepseek_1000_expressive5" / "table4_lambda_validation.csv"
    rows = _read_csv(fp)
    keymap = {
        "LightGCN + Raw Profile": "Profile-Raw",
        "LightGCN + Remove Repair": "Profile-Remove",
        "LightGCN + Evidence-Weighted Repair": "Profile-EGPR",
    }
    out = {v: {"lambda": [], "ndcg": []} for v in keymap.values()}
    for r in rows:
        m = keymap.get(r["Method"])
        if m is None:
            continue
        out[m]["lambda"].append(float(r["Lambda"]))
        out[m]["ndcg"].append(float(r["NDCG@20"]))
    return out


def _read_csv(fp):
    import csv
    with open(fp) as f:
        return list(csv.DictReader(f))


def draw_lambda_sensitivity():
    data = {
        "ML-1M (top-50)": _load_ml1m_lambda(),
        "Yelp (top-100)": _load_yelp_lambda("100"),
        "Amazon Books (top-100)": _load_amazon_lambda(),
    }

    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.3))
    for ax_i, (ds, methods) in enumerate(data.items()):
        ax = axes[ax_i]
        base = DATASET_BASE[ds]
        # Sort lambdas; data already sorted but ensure
        all_ndcg = []
        for m, series in methods.items():
            pairs = sorted(zip(series["lambda"], series["ndcg"]))
            xs = [p[0] for p in pairs]
            ys = [p[1] for p in pairs]
            all_ndcg.extend(ys)
            ax.plot(xs, ys,
                    marker=METHOD_MARKER[m], color=METHOD_COLOR[m],
                    linewidth=1.5, markersize=6, label=m,
                    markeredgecolor="black", markeredgewidth=0.5)
        ax.axhline(base, color="#666", linewidth=0.8, linestyle="--", alpha=0.7,
                   label="Base (no LLM)" if ax_i == 0 else None)
        ax.set_title(ds, fontsize=10, weight="bold")
        ax.set_xlabel(r"Mixing weight $\lambda$")
        if ax_i == 0:
            ax.set_ylabel("NDCG@20")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(linestyle=":", alpha=0.4)
        # Highlight selected lambda
        sel = {"ML-1M (top-50)": 0.5, "Yelp (top-100)": 0.5, "Amazon Books (top-100)": 0.5}[ds]
        ax.axvline(sel, color="#aaa", linewidth=0.6, linestyle=":", alpha=0.7)
        ymin, ymax = min(min(all_ndcg), base), max(all_ndcg)
        pad = (ymax - ymin) * 0.15 + 1e-5
        ax.set_ylim(ymin - pad, ymax + pad)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.02), frameon=False)
    fig.suptitle(r"Validation NDCG@20 as a function of mixing weight $\lambda$",
                 fontsize=11, weight="bold")
    fig.tight_layout(rect=[0, 0.06, 1, 0.94])
    out = HERE / "lambda_sensitivity.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


# -------------------------------------------------------------------------
# Tau sensitivity (claim coverage analysis)
# -------------------------------------------------------------------------

def _load_claim_supports(jsonl_path):
    scores = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            s = row.get("support_score")
            if s is not None:
                scores.append(float(s))
    return np.asarray(scores)


def draw_tau_sensitivity():
    datasets = {
        "ML-1M": RESULTS_ROOT / "ml1m_seed42_deepseek_2000_expressive5" / "claim_support.jsonl",
        "Yelp": RESULTS_ROOT / "yelp_seed42_deepseek_1000_expressive5" / "claim_support.jsonl",
        "Amazon Books": RESULTS_ROOT / "amazon_books_seed42_deepseek_1000_expressive5" / "claim_support.jsonl",
    }
    dataset_color = {"ML-1M": "#1f77b4", "Yelp": "#d62728", "Amazon Books": "#2ca02c"}

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))

    # Panel (a): claim coverage histogram per dataset
    ax = axes[0]
    bins = np.linspace(0, 1, 26)
    for ds, fp in datasets.items():
        scores = _load_claim_supports(fp)
        ax.hist(scores, bins=bins, alpha=0.45, color=dataset_color[ds],
                edgecolor=dataset_color[ds], linewidth=0.6,
                label=f"{ds} ($n={len(scores):,}$)", density=True)
    ax.axvline(0.45, color="#000", linewidth=1.2, linestyle="--",
               label=r"Selected $\tau=0.45$")
    ax.set_xlabel(r"Per-claim evidence coverage $\mathrm{cov}(c, E_u)$")
    ax.set_ylabel("Density")
    ax.set_title("(a) Claim coverage distribution",
                 fontsize=10, weight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", fontsize=7.5)
    ax.grid(linestyle=":", alpha=0.4)

    # Panel (b): UCR(τ) curve per dataset
    ax = axes[1]
    taus = np.linspace(0.0, 1.0, 51)
    for ds, fp in datasets.items():
        scores = _load_claim_supports(fp)
        ucr = np.array([np.mean(scores < t) for t in taus])
        ax.plot(taus, ucr, color=dataset_color[ds], linewidth=1.8,
                label=ds)
    ax.axvline(0.45, color="#000", linewidth=1.2, linestyle="--",
               label=r"Selected $\tau=0.45$")
    ax.set_xlabel(r"Coverage threshold $\tau$")
    ax.set_ylabel(r"UCR($\tau$) = fraction of claims removed")
    ax.set_title(r"(b) Unsupported-claim rate as a function of $\tau$",
                 fontsize=10, weight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(linestyle=":", alpha=0.4)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    fig.suptitle(r"Sensitivity of evidence governance to the coverage threshold $\tau$",
                 fontsize=11, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = HERE / "tau_sensitivity.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    draw_lambda_sensitivity()
    draw_tau_sensitivity()

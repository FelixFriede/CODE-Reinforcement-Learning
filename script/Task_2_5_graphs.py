# script/Task_2_5_graphs.py

# general
from __future__ import annotations
import os, json
import numpy as np
import matplotlib.pyplot as plt
from typing import Any, Dict, List

# global utilities
from util.io_helpers import OUT_DIR, _ensure_dir


# -----------------------------
# local utilities
# -----------------------------

def _ci95_from_var(var: np.ndarray, N: int) -> np.ndarray:
    # 95% CI half-width for mean using normal approx: 1.96 * sqrt(var / N)
    return 1.96 * np.sqrt(var / float(N))


# -----------------------------
# Plotting
# -----------------------------

def plot_regret_with_ci(results: List[Dict[str, Any]], out_dir: str, kind: str = "cum"):
    """
    kind: "cum" or "inst"
    """
    _ensure_dir(out_dir)
    n_steps = results[0]["n_steps"]
    N = results[0]["N"]
    t = np.arange(n_steps)

    plt.figure()
    for r in results:
        if kind == "cum":
            mean = r["cum_regret_mean"]
            var = r["cum_regret_var"]
            ylabel = "Average cumulative regret"
            title = "Cumulative regret with 95% CI"
            fname = "regret_cumulative_ci.png"
        else:
            mean = r["inst_regret_mean"]
            var = r["inst_regret_var"]
            ylabel = "Average instant regret"
            title = "Instant regret with 95% CI"
            fname = "regret_instant_ci.png"

        ci = _ci95_from_var(var, N)
        plt.plot(t, mean, label=f"{r['name']} {r['params']}")
        plt.fill_between(t, mean - ci, mean + ci, alpha=0.15)

    plt.xlabel("t")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(ncol=2, fontsize=8)
    plt.savefig(os.path.join(out_dir, fname), dpi=160)
    plt.close()


def plot_boxplot_true_vs_estimates(results: List[Dict[str, Any]], out_dir: str):
    """
    Rank-aligned boxplots (best arm rank 1 .. K):
      - true m_eans
      - each algorithm's final estimates (if available; NaNs are ignored via masked arrays)
    """
    _ensure_dir(out_dir)
    K = results[0]["K"]

    true_ranked = results[0]["true_means_ranked"]  # (N,K) same across all results
    data = []
    labels = []

    # true m_eans first
    for r in range(K):
        data.append(true_ranked[:, r])
        labels.append(f"true r{r+1}")

    # then each algorithm estimates
    for res in results:
        est = res["est_means_ranked_final"]  # (N,K)
        for r in range(K):
            col = est[:, r]
            # ignore NaNs cleanly
            col = col[np.isfinite(col)]
            data.append(col)
            labels.append(f"{res['name']} r{r+1}")

    plt.figure(figsize=(max(10, 0.22 * len(labels)), 5))
    plt.boxplot(data, showfliers=False)
    plt.xticks(np.arange(1, len(labels) + 1), labels, rotation=90)
    plt.ylabel("Value")
    plt.title("True means vs final estimates (rank-aligned)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "box_true_vs_estimates.png"), dpi=160)
    plt.close()


def plot_boxplot_play_probabilities(results: List[Dict[str, Any]], out_dir: str):
    """
    Rank-aligned arm play probabilities over the full horizon (counts/n_steps).
    """
    _ensure_dir(out_dir)
    K = results[0]["K"]

    data = []
    labels = []

    for res in results:
        probs = res["play_prob_ranked"]  # (N,K)
        for r in range(K):
            data.append(probs[:, r])
            labels.append(f"{res['name']} r{r+1}")

    plt.figure(figsize=(max(10, 0.22 * len(labels)), 5))
    plt.boxplot(data, showfliers=False)
    plt.xticks(np.arange(1, len(labels) + 1), labels, rotation=90)
    plt.ylabel("Probability")
    plt.title("Play probabilities per arm rank (counts/n)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "box_play_probabilities.png"), dpi=160)
    plt.close()


def plot_boxplot_regrets(results: List[Dict[str, Any]], out_dir: str, kind: str):
    """
    Boxplot of final cumulative regrets (one value per run).
    """
    _ensure_dir(out_dir)

    data = [res["final_regret_per_run"] for res in results]
    labels = [f"{res['name']} {res['params']}" for res in results]

    plt.figure(figsize=(max(10, 0.35 * len(labels)), 5))
    plt.boxplot(data, showfliers=False)
    plt.xticks(np.arange(1, len(labels) + 1), labels, rotation=90)
    plt.ylabel("Cumulative regret at n")
    plt.title("Final regret distributions")
    plt.tight_layout()
    
    plt.savefig(os.path.join(out_dir, "box_final_regrets.png"), dpi=160)


    plt.close()


# -----------------------------
# data loader
# -----------------------------

def _load_run(exp_dir: str) -> Dict[str, Any]:
    # meta
    with open(os.path.join(exp_dir, "result_meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)

    # arrays
    arrays_path = os.path.join(exp_dir, "result_arrays.npz")
    with np.load(arrays_path, allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}

    return {**meta, **arrays}

# -----------------------------
# Main
# -----------------------------

def main():
    out_dir = _ensure_dir(os.path.join(OUT_DIR, "ex2_all_algorithms_bernoulli"))

    # Select which algorithms to include (comment out any you don't want)
    SELECTED_ALGOS = [
        "ETC",
        "Greedy",
        "EpsGreedyFixed",
        "EpsGreedyDecreasing",
        "UCB",
        "UCBSubGaussian",
        "BoltzmannSoftmax",
        "BoltzmannGumbel",
        "BoltzmannArbitraryNoise(gumbel)",
        "GumbelScaledBonus",
        "PolicyGradient",
        "PolicyGradientBaseline",
    ]

    def _algo_name(r: Dict[str, Any]) -> str:
        return r.get("name") or r.get("algo_name") or ""

    # Load index
    index_path = os.path.join(out_dir, "index.json")
    with open(index_path, "r", encoding="utf-8") as f:
        index = json.load(f)

    # Load only selected runs (in the order given above)
    index_by_algo = {e["algorithm"]: e for e in index}
    results: List[Dict[str, Any]] = []
    for algo in SELECTED_ALGOS:
        entry = index_by_algo.get(algo)
        if entry is None:
            print(f"Warning: algorithm '{algo}' not found in {index_path}; skipping.")
            continue
        exp_dir = os.path.join(out_dir, entry["dir"])
        results.append(_load_run(exp_dir))

    if not results:
        raise RuntimeError("No results loaded. Check SELECTED_ALGOS and index.json.")

    # --- Requested plots ---
    plot_regret_with_ci(results, out_dir=out_dir, kind="cum")
    plot_regret_with_ci(results, out_dir=out_dir, kind="inst")

    plot_boxplot_true_vs_estimates(results, out_dir=out_dir)       # (i)
    plot_boxplot_play_probabilities(results, out_dir=out_dir)      # (ii)
    plot_boxplot_regrets(results, out_dir=out_dir, kind="normal")  # (iii)

if __name__ == "__main__":
    main()
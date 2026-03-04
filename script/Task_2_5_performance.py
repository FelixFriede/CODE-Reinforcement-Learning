# experiments/ex2_all_algorithms_bernoulli.py
#
# Follows the general structure of the attached Exercise Sheet 1 solution (bulk-vectorized N runs,
# per-step mean/var tracking, plotting helpers). :contentReference[oaicite:0]{index=0}
#
# Task:
# - 5-armed Bernoulli bandit with random m_eans (shared across all algorithms for fairness)
# - horizon n=10_000
# - average over N=1_000 runs (vectorized as Gang_of_Bandits with n_bandits=N)
# - compute-efficient parameter tuning via coarse grid search on shorter horizon
# - plots:
#   (a) regrets over time with 95% CI shading
#   (b) boxplots at time n:
#       i) true m_eans vs algorithm estimates (rank-aligned per run)
#      ii) probability of playing each arm (counts/n)
#     iii) cumulative regret distribution per algorithm

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

from src.bandits import Gang_of_Bandits

# bulk algorithms
from src.etc import ETCBulkAlgorithm

from src.greedy import (
    GreedyBulkAlgorithm,
    EpsilonGreedyFixedBulkAlgorithm,
    EpsilonGreedyDecreasingBulkAlgorithm,
)

from src.ucb import UCBulkAlgorithm, UCBSubGaussianBulkAlgorithm

from src.boltzmann import (
    BoltzmannExplorationBulkAlgorithm,
    BoltzmannGumbelTrickBulkAlgorithm,
    BoltzmannArbitraryNoiseBulkAlgorithm,
    GumbelScaledBonusBulkAlgorithm,
)

from src.gradient import (
    PolicyGradientBulkAlgorithm,
    PolicyGradientBaselineBulkAlgorithm,
)

from util.io_helpers import OUT_DIR


# -----------------------------
# Small utilities
# -----------------------------

def _mean_var(x, axis=0):
    x = np.asarray(x)
    return x.mean(axis=axis), x.var(axis=axis)  # population variance


def _ci95_from_var(var: np.ndarray, N: int) -> np.ndarray:
    # 95% CI half-width for mean using normal approx: 1.96 * sqrt(var / N)
    return 1.96 * np.sqrt(var / float(N))


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _softmax_rowwise(theta: np.ndarray) -> np.ndarray:
    # stable softmax for diagnostics only (not required for algorithms)
    x = theta - np.max(theta, axis=1, keepdims=True)
    ex = np.exp(x)
    s = np.sum(ex, axis=1, keepdims=True)
    return ex / s


# -----------------------------
# Algorithm registry
# -----------------------------

@dataclass(frozen=True)
class AlgoSpec:
    name: str
    # factory: (bandit: Gang_of_Bandits, params: dict) -> algorithm instance
    factory: Callable[[Gang_of_Bandits, Dict[str, Any]], Any]
    # parameter grid for tuning (list of dicts)
    grid: List[Dict[str, Any]]


def _algo_specs() -> List[AlgoSpec]:
    return [
        # ETC
        AlgoSpec(
            name="ETC",
            factory=lambda b, p: ETCBulkAlgorithm(b, exploration_rounds=p["m"]),
            grid=[{"m": m} for m in [1, 2, 3, 5, 10, 20, 50, 100]],
        ),
        # Greedy family
        AlgoSpec(
            name="Greedy",
            factory=lambda b, p: GreedyBulkAlgorithm(b),
            grid=[{}],
        ),
        AlgoSpec(
            name="EpsGreedyFixed",
            factory=lambda b, p: EpsilonGreedyFixedBulkAlgorithm(b, epsilon=p["epsilon"]),
            grid=[{"epsilon": e} for e in [0.01, 0.03, 0.05, 0.1, 0.2]],
        ),
        AlgoSpec(
            name="EpsGreedyDecreasing",
            factory=lambda b, p: EpsilonGreedyDecreasingBulkAlgorithm(b, epsilon0=p["epsilon0"]),
            grid=[{"epsilon0": e0} for e0 in [0.1, 0.3, 0.5, 1.0, 2.0, 5.0]],
        ),
        # UCB
        AlgoSpec(
            name="UCB",
            factory=lambda b, p: UCBulkAlgorithm(b, delta=p["delta"]),
            grid=[{"delta": d} for d in [0.01, 0.03, 0.05, 0.1, 0.2]],
        ),
        # For Bernoulli in [0,1], a safe subgaussian proxy is sigma=0.5
        AlgoSpec(
            name="UCBSubGaussian",
            factory=lambda b, p: UCBSubGaussianBulkAlgorithm(b, delta=p["delta"], sigma=0.5),
            grid=[{"delta": d} for d in [0.01, 0.03, 0.05, 0.1, 0.2]],
        ),
        # Boltzmann family
        AlgoSpec(
            name="BoltzmannSoftmax",
            factory=lambda b, p: BoltzmannExplorationBulkAlgorithm(b, theta=p["theta"]),
            grid=[{"theta": th} for th in [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]],
        ),
        AlgoSpec(
            name="BoltzmannGumbel",
            factory=lambda b, p: BoltzmannGumbelTrickBulkAlgorithm(b, theta=p["theta"]),
            grid=[{"theta": th} for th in [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]],
        ),
        AlgoSpec(
            name="BoltzmannArbitraryNoise(gumbel)",
            factory=lambda b, p: BoltzmannArbitraryNoiseBulkAlgorithm(b, theta=p["theta"], noise="gumbel"),
            grid=[{"theta": th} for th in [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]],
        ),
        AlgoSpec(
            name="GumbelScaledBonus",
            factory=lambda b, p: GumbelScaledBonusBulkAlgorithm(b, C=p["C"]),
            grid=[{"C": c} for c in [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]],
        ),
        # Policy gradient
        AlgoSpec(
            name="PolicyGradient",
            factory=lambda b, p: PolicyGradientBulkAlgorithm(b, alpha=p["alpha"]),
            grid=[{"alpha": a} for a in [0.01, 0.03, 0.05, 0.1, 0.2]],
        ),
        AlgoSpec(
            name="PolicyGradientBaseline",
            factory=lambda b, p: PolicyGradientBaselineBulkAlgorithm(b, alpha=p["alpha"]),
            grid=[{"alpha": a} for a in [0.01, 0.03, 0.05, 0.1, 0.2]],
        ),
    ]


# -----------------------------
# Core runner (bulk, vectorized)
# -----------------------------

def run_bulk_experiment(
    spec: AlgoSpec,
    params: Dict[str, Any],
    means: np.ndarray,
    n_steps: int,
    seed: int | None = None,
) -> Dict[str, Any]:
    """
    Run N independent experiments in parallel (N = m_eans.shape[0]) for n_steps.

    Returns dict with:
      - inst_regret_mean/var (t)
      - cum_regret_mean/var (t)
      - final_regret_per_run (N,)
      - true_m_eans_ranked (N,K)
      - est_m_eans_ranked (N,K)  (final)
      - play_prob_per_run_ranked (N,K)  (counts/n_steps, rank-aligned by true m_eans)
    """
    if seed is not None:
        np.random.seed(seed)

    N, K = means.shape

    gang = Gang_of_Bandits(
        n_bandits=N,
        n_arms=K,
        distribution="bernoulli",
        means=means,
    )
    algo = spec.factory(gang, params)

    rows = np.arange(N)
    true_best = np.argmax(gang.means, axis=1)

    # rank alignment per run (best -> worst)
    order = np.argsort(gang.means, axis=1)[:, ::-1]
    true_means_ranked = np.take_along_axis(gang.means, order, axis=1)

    inst_regret_mean = np.empty(n_steps, dtype=float)
    inst_regret_var = np.empty(n_steps, dtype=float)
    cum_regret_mean = np.empty(n_steps, dtype=float)
    cum_regret_var = np.empty(n_steps, dtype=float)

    cum_regret_per_run = np.zeros(N, dtype=float)

    # counts for play probabilities over the full horizon
    play_counts = np.zeros((N, K), dtype=np.int32)

    for t in range(n_steps):
        if t % 500 == 0:
            sys.stdout.write(f"\rRunning {spec.name:<28} | {params} | step {t}/{n_steps}")
            sys.stdout.flush()

        chosen, rewards = algo.step()  # chosen: (N,), rewards: (N,)
        play_counts[rows, chosen] += 1

        mu_star = gang.means[rows, true_best]
        mu_chosen = gang.means[rows, chosen]
        inst_regret = mu_star - mu_chosen  # (N,)

        cum_regret_per_run += inst_regret

        inst_regret_mean[t], inst_regret_var[t] = _mean_var(inst_regret, axis=0)
        cum_regret_mean[t], cum_regret_var[t] = _mean_var(cum_regret_per_run, axis=0)

    sys.stdout.write("\r\033[K")
    sys.stdout.flush()

    # final estimates extraction (handle different internal state names)
    if hasattr(algo, "arm_reward_sums") and hasattr(algo, "arm_pull_counts"):
        denom = np.maximum(algo.arm_pull_counts, 1)
        est = algo.arm_reward_sums / denom
    elif hasattr(algo, "arm_value_estimates"):
        est = np.asarray(algo.arm_value_estimates)
    elif hasattr(algo, "theta"):
        # policy gradient: interpret softmax(theta) as preference; not a mean estimate.
        # For plotting "m_eans vs estimates", we use empirical m_eans from tracking if present;
        # if absent, we use NaNs.
        est = np.full((N, K), np.nan, dtype=float)
    else:
        est = np.full((N, K), np.nan, dtype=float)

    est_means_ranked = np.take_along_axis(est, order, axis=1)

    play_prob_per_run = play_counts.astype(float) / float(n_steps)
    play_prob_per_run_ranked = np.take_along_axis(play_prob_per_run, order, axis=1)

    return {
        "name": spec.name,
        "params": params,
        "n_steps": n_steps,
        "N": N,
        "K": K,
        "true_means_ranked": true_means_ranked,              # (N,K)
        "est_means_ranked_final": est_means_ranked,          # (N,K)
        "play_prob_ranked": play_prob_per_run_ranked,        # (N,K)
        "inst_regret_mean": inst_regret_mean,
        "inst_regret_var": inst_regret_var,
        "cum_regret_mean": cum_regret_mean,
        "cum_regret_var": cum_regret_var,
        "final_regret_per_run": cum_regret_per_run.copy(),   # (N,)
    }


def tune_parameters(
    spec: AlgoSpec,
    means: np.ndarray,
    tune_steps: int = 3_000,
    seed: int = 123,
) -> Tuple[Dict[str, Any], float]:
    """
    Coarse grid search: pick params minimizing mean cumulative regret at tune_steps.
    (Compute-efficient because each candidate runs in bulk over N=1000 in one go.)
    """
    best_params = spec.grid[0]
    best_score = float("inf")

    for i, p in enumerate(spec.grid):
        res = run_bulk_experiment(spec, p, means=means, n_steps=tune_steps, seed=seed + i)
        score = float(res["cum_regret_mean"][-1])  # mean cumulative regret at horizon
        if score < best_score:
            best_score = score
            best_params = p

    return best_params, best_score


# -----------------------------
# Plotting (exercise-sheet style)
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


def plot_boxplot_regrets(results: List[Dict[str, Any]], out_dir: str):
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
# Main
# -----------------------------

def main():
    # experiment constants
    K = 5
    n_steps = 10_000
    N = 1_000

    # tuning horizon (keep smaller for compute efficiency)
    tune_steps = 3_000

    # top-level output folder
    out_dir = _ensure_dir(os.path.join(OUT_DIR, "ex2_all_algorithms_bernoulli"))
    print(f"Writing outputs to: {out_dir}")

    # Make ONE shared random m_eans matrix for fairness across algorithms.
    # (Each row is one independent bandit instance.)
    np.random.seed(0)
    means = np.random.rand(N, K)

    # Tune parameters (coarse grid) + run full horizon with best params
    results: List[Dict[str, Any]] = []

    for spec in _algo_specs():
        print(f"\nTuning {spec.name} over {len(spec.grid)} candidate(s) ...")
        best_params, best_score = tune_parameters(spec, means=means, tune_steps=tune_steps, seed=1000)
        print(f"Best params for {spec.name}: {best_params} (tune mean regret @ {tune_steps} = {best_score:.4f})")

        print(f"Running full horizon for {spec.name} ...")
        res = run_bulk_experiment(spec, best_params, means=means, n_steps=n_steps, seed=2000)
        results.append(res)
        print(f"Done {spec.name}: mean final regret = {float(res['cum_regret_mean'][-1]):.4f}")

    # Plots requested:
    # (a) regret curves with 95% CI shading
    plot_regret_with_ci(results, out_dir=out_dir, kind="cum")
    plot_regret_with_ci(results, out_dir=out_dir, kind="inst")

    # (b) boxplots at end of horizon
    plot_boxplot_true_vs_estimates(results, out_dir=out_dir)      # (i)
    plot_boxplot_play_probabilities(results, out_dir=out_dir)     # (ii)
    plot_boxplot_regrets(results, out_dir=out_dir)                # (iii)

    print("\nAll done.")
    print("Created:")
    print(" - regret_cumulative_ci.png")
    print(" - regret_instant_ci.png")
    print(" - box_true_vs_estimates.png")
    print(" - box_play_probabilities.png")
    print(" - box_final_regrets.png")


if __name__ == "__main__":
    main()
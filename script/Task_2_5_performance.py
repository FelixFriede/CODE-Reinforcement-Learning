# experiments/ex2_all_algorithms_bernoulli.py
# This file is CORE.
# However, data saving (instead of straight image rendering) was introduced late and is purely AI generated and not proof read.


# general
from __future__ import annotations
import os, sys, json
import numpy as np
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple

# bandit supporting bulk pull
from src.bandits import Gang_of_Bandits

# bulk algorithms
from src.etc import ETCBulkAlgorithm
from src.greedy import GreedyBulkAlgorithm, EpsilonGreedyFixedBulkAlgorithm, EpsilonGreedyDecreasingBulkAlgorithm
from src.ucb import UCBulkAlgorithm, UCBSubGaussianBulkAlgorithm
from src.boltzmann import BoltzmannExplorationBulkAlgorithm, BoltzmannGumbelTrickBulkAlgorithm, BoltzmannArbitraryNoiseBulkAlgorithm, GumbelScaledBonusBulkAlgorithm
from src.gradient import PolicyGradientBulkAlgorithm, PolicyGradientBaselineBulkAlgorithm

# global utilities
from util.io_helpers import OUT_DIR, _ensure_dir


# -----------------------------
# local utilities
# -----------------------------

def _mean_var(x, axis=0):
    x = np.asarray(x)
    return x.mean(axis=axis), x.var(axis=axis)  # population variance

# BUG: Why is this not being used?
def _softmax_rowwise(theta: np.ndarray) -> np.ndarray:
    # stable softmax for diagnostics only (not required for algorithms)
    x = theta - np.max(theta, axis=1, keepdims=True)
    ex = np.exp(x)
    s = np.sum(ex, axis=1, keepdims=True)
    return ex / s


# ----------------------
# data saving helpers
# ----------------------

def _split_res(res: Dict[str, Any]):
    arrays = {}
    meta = {}

    for k, v in res.items():
        if isinstance(v, np.ndarray):
            arrays[k] = v
        else:
            meta[k] = v

    return arrays, meta


def _save_run(out_dir: str, res: Dict[str, Any]):
    arrays, meta = _split_res(res)

    np.savez_compressed(os.path.join(out_dir, "result_arrays.npz"), **arrays)

    with open(os.path.join(out_dir, "result_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)


# -----------------------------
# Algorithm registry
# -----------------------------

@dataclass(frozen=True)
class AlgoSpec:
    name: str
    factory: Callable[[Gang_of_Bandits, Dict[str, Any]], Any] # (bandit: Gang_of_Bandits, params: dict) -> algorithm instance
    grid: List[Dict[str, Any]] # parameter grid for tuning (list of dicts)

# BUG: For some algorithms the parameter edge case is optimal. Change parameter range.
def _algo_specs() -> List[AlgoSpec]:

    def linspace_params(lo, hi, n=100, *, key: str, cast=None, round_to: int | None = None):
        # Evenly spaced helper (inclusive endpoints), with optional rounding and int-casting.
        vals = np.linspace(lo, hi, n)
        out = []
        for v in vals:
            if round_to is not None:
                v = float(np.round(v, round_to))
            if cast is not None:
                v = cast(v)
            out.append({key: v})
        return out

    # Comment out Algorithms to remove them from data collection. Make sure data files are named appropriatly.
    return [
        # ETC
        AlgoSpec(name="ETC", factory=lambda b, p: ETCBulkAlgorithm(b, exploration_rounds=p["m"]), grid=linspace_params(1, 50, 50, key="m", cast=int)),
        
        # Greedy family
        AlgoSpec(name="Greedy", factory=lambda b, p: GreedyBulkAlgorithm(b), grid=[{}]),
        AlgoSpec(name="EpsGreedyFixed", factory=lambda b, p: EpsilonGreedyFixedBulkAlgorithm(b, epsilon=p["epsilon"]), grid=linspace_params(0.01, 0.5, 50, key="epsilon", round_to=6)),
        AlgoSpec(name="EpsGreedyDecreasing", factory=lambda b, p: EpsilonGreedyDecreasingBulkAlgorithm(b, epsilon0=p["epsilon0"]), grid=linspace_params(0.10, 50.00, 50, key="epsilon0", round_to=6)),
        
        # UCB family
        AlgoSpec(name="UCB", factory=lambda b, p: UCBulkAlgorithm(b, delta=p["delta"]), grid=linspace_params(0.01, 0.5, 50, key="delta", round_to=6)),
        AlgoSpec(name="UCBSubGaussian", factory=lambda b, p: UCBSubGaussianBulkAlgorithm(b, delta=p["delta"], sigma=0.5), grid=linspace_params(0.01, 0.5, 50, key="delta", round_to=6)),
        
        # Boltzmann family
        AlgoSpec(name="BoltzmannSoftmax", factory=lambda b, p: BoltzmannExplorationBulkAlgorithm(b, theta=p["theta"]), grid=linspace_params(0.10, 50.00, 50, key="theta", round_to=6)),
        AlgoSpec(name="BoltzmannGumbel", factory=lambda b, p: BoltzmannGumbelTrickBulkAlgorithm(b, theta=p["theta"]), grid=linspace_params(0.10, 50.00, 50, key="theta", round_to=6)),
        AlgoSpec(name="BoltzmannArbitraryNoise(gumbel)", factory=lambda b, p: BoltzmannArbitraryNoiseBulkAlgorithm(b, theta=p["theta"], noise="gumbel"), grid=linspace_params(0.10, 50.00, 50, key="theta", round_to=6)),
        AlgoSpec(name="GumbelScaledBonus", factory=lambda b, p: GumbelScaledBonusBulkAlgorithm(b, C=p["C"]), grid=linspace_params(0.01, 0.50, 50, key="C", round_to=6)),
        
        # Policy gradient
        AlgoSpec(name="PolicyGradient", factory=lambda b, p: PolicyGradientBulkAlgorithm(b, alpha=p["alpha"]), grid=linspace_params(0.01, 0.50, 50, key="alpha", round_to=6)),
        AlgoSpec(name="PolicyGradientBaseline", factory=lambda b, p: PolicyGradientBaselineBulkAlgorithm(b, alpha=p["alpha"]), grid=linspace_params(0.01, 0.50, 50, key="alpha", round_to=6)),
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
    Run N independent experiments in parallel (N = means.shape[0]) for n_steps.

    Returns dict with:
      - inst_regret_mean/var (t)
      - cum_regret_mean/var (t)
      - final_regret_per_run (N,)
      - true_means_ranked (N,K)
      - est_means_ranked (N,K)  (final)
      - play_prob_per_run_ranked (N,K)  (counts/n_steps, rank-aligned by true means)
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
        if t % 1000 == 0:
            sys.stdout.write(f"\rRunning {spec.name:<28} | {params} | gang size N={N} | step {t}/{n_steps}")
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
        # For plotting "means vs estimates", we use empirical means from tracking if present;
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


# -----------------------------
# Tuning: successive 1/3-ing with reduced resources
# -----------------------------

def tune_parameters(
    spec: AlgoSpec,
    means: np.ndarray,
    seed: int = 123,
) -> Tuple[Dict[str, Any], float]:
    # Successive halving over the parameter grid.

    # If only one candidate (e.g., Greedy), just sample once at full step length.
    if len(spec.grid) == 1:
        p = spec.grid[0]
        means_r = means[: min(500, len(means))]
        res = run_bulk_experiment(spec, p, means=means_r, n_steps=10000, seed=seed)
        score = float(res["cum_regret_mean"][-1])
        return p, score

    eta = 3  # halving factor
    n_rounds = 3 # 50 -> 16 -> 5
    candidates = list(spec.grid) 

    # Hard-coded increasing budgets per round
    steps_schedule = [1000, 3000, 10000]
    N_schedule = [25, 50, 100]

    best_params = candidates[0]
    best_score = float("inf")

    for r in range(n_rounds):
        n_steps_r = steps_schedule[r]
        N_r = N_schedule[r]

        # Truncate means to control "bulk N"
        means_r = means[: min(N_r, len(means))]

        scored: List[Tuple[float, Dict[str, Any]]] = []
        for i, p in enumerate(candidates):
            res = run_bulk_experiment(
                spec,
                p,
                means=means_r,
                n_steps=n_steps_r,
                seed=seed + 10_000 * r + i,
            )
            score = float(res["cum_regret_mean"][-1])
            scored.append((score, p))

        scored.sort(key=lambda x: x[0])  # lower regret is better

        # Track best seen
        if scored[0][0] < best_score:
            best_score = scored[0][0]
            best_params = scored[0][1]

        if r == n_rounds - 1:
            break

        # Keep top fraction
        keep = max(1, len(scored) // eta)
        candidates = [p for _, p in scored[:keep]]

    print(f"Best params for {spec.name}: {best_params} (tune mean regret @ {10_000} steps = {best_score:.4f})")

    return best_params


# -----------------
# main
# -----------------

def main():
    # experiment constants
    K = 5
    n_steps = 10_000
    N = 1_000

    # shared bandit instances
    np.random.seed(0)
    means = np.random.rand(N, K)

    base_dir = _ensure_dir(os.path.join(OUT_DIR, "ex2_all_algorithms_bernoulli"))
    print(f"Writing data to: {base_dir}")

    index = []

    for spec in _algo_specs():
        print(f"\r\033[KRunning algorithm: {spec.name}", end="")

        # parameter tuning
        best_params = tune_parameters(spec, means=means, seed=1000)

        # full run
        res = run_bulk_experiment(spec, best_params, means=means, n_steps=n_steps, seed=2000)
        print(f"Done {spec.name}: mean final regret = {float(res['cum_regret_mean'][-1]):.4f}")

        # algorithm-specific folder
        algo_dir = os.path.join(base_dir, spec.name)
        os.makedirs(algo_dir, exist_ok=True)

        _save_run(algo_dir, res)

        index.append({
            "algorithm": spec.name,
            "dir": spec.name
        })

        print(f"\r\033[KSaved results for algorithm: {spec.name}")

    # save lightweight index
    with open(os.path.join(base_dir, "index.json"), "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)


if __name__ == "__main__":
    main()
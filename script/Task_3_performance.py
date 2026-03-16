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
from src.greedy import EpsilonGreedyFixedBulkAlgorithm, EpsilonGreedyDecreasingBulkAlgorithm
from src.ucb import UCBBulkAlgorithm
from src.boltzmann import BoltzmannExplorationBulkAlgorithm
from src.gradient import PolicyGradientBulkAlgorithm

# global utilities
from util.io_helpers import OUT_DIR, _ensure_dir


# -----------------------------
# local utilities
# -----------------------------

def _mean_var(x, axis=0):
    x = np.asarray(x)
    return x.mean(axis=axis), x.var(axis=axis)  # population variance


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
    plotgrid: List

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

    return [
        # ETC
        AlgoSpec(name="ETC",factory=lambda b, p: ETCBulkAlgorithm(b, exploration_rounds=p["m"]),grid=linspace_params(6, 30, 25, key="m", cast=int),plotgrid=[10,15,20,25,30,35,40,45,50],),
        # Greedy family
        AlgoSpec(name="EpsGrdy",factory=lambda b, p: EpsilonGreedyFixedBulkAlgorithm(b, epsilon=p["epsilon"]),grid=linspace_params(0.002, 0.10, 50, key="epsilon", round_to=6),plotgrid=[0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1],),
        AlgoSpec(name="Eps0Grdy",factory=lambda b, p: EpsilonGreedyDecreasingBulkAlgorithm(b, epsilon0=p["epsilon0"]),grid=linspace_params(1, 50, 50, key="epsilon0", round_to=6),plotgrid=[4, 6, 8, 10, 12, 14, 16, 18, 20, 22],),
        # UCB family
        AlgoSpec(name="UCB",factory=lambda b, p: UCBBulkAlgorithm(b, delta=p["delta"]),grid=linspace_params(0.02, 1, 50, key="delta", round_to=6),plotgrid=[0.2,0.3,0.4,0.5,0.6,0.7,0.8],),
        # Boltzmann family
        AlgoSpec(name="BltzSM",factory=lambda b, p: BoltzmannExplorationBulkAlgorithm(b, theta=p["theta"]),grid=linspace_params(1, 20.6, 50, key="theta", round_to=6),plotgrid=[4,6,8,10,12,14,16,18,20,22],),
        # Policy gradient
        AlgoSpec(name="PG",factory=lambda b, p: PolicyGradientBulkAlgorithm(b, alpha=p["alpha"]),grid=linspace_params(0.01, 0.50, 50, key="alpha", round_to=6),plotgrid=[0.05, 0.15, 0.25, 0.35, 0.45, 0.55],),
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
      - optimal_action_mean/var (t)
      - cum_optimal_action_mean/var (t)
      - optimal_action_count_per_run (N,)
      - optimal_action_share_per_run (N,)
      - final_estimation_mse_per_run (N,)
      - final_regret_per_run (N,)
      - true_means_ranked (N,K)
      - est_means_ranked_final (N,K)
      - play_prob_ranked (N,K)
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

    optimal_action_mean = np.empty(n_steps, dtype=float)
    optimal_action_var = np.empty(n_steps, dtype=float)
    cum_optimal_action_mean = np.empty(n_steps, dtype=float)
    cum_optimal_action_var = np.empty(n_steps, dtype=float)

    cum_regret_per_run = np.zeros(N, dtype=float)
    optimal_action_count_per_run = np.zeros(N, dtype=np.int32)

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

        optimal_action = (chosen == true_best).astype(float)  # (N,)
        optimal_action_count_per_run += optimal_action.astype(np.int32)

        inst_regret_mean[t], inst_regret_var[t] = _mean_var(inst_regret, axis=0)
        cum_regret_mean[t], cum_regret_var[t] = _mean_var(cum_regret_per_run, axis=0)

        optimal_action_mean[t], optimal_action_var[t] = _mean_var(optimal_action, axis=0)
        cum_optimal_action_mean[t], cum_optimal_action_var[t] = _mean_var(optimal_action_count_per_run, axis=0)

    sys.stdout.write("\r\033[K")
    sys.stdout.flush()

    # final estimates extraction (handle different internal state names)
    if hasattr(algo, "arm_reward_sums") and hasattr(algo, "arm_pull_counts"):
        denom = np.maximum(algo.arm_pull_counts, 1)
        est = algo.arm_reward_sums / denom
    elif hasattr(algo, "arm_value_estimates"):
        est = np.asarray(algo.arm_value_estimates)
    elif hasattr(algo, "theta"):
        # policy gradient: not a mean estimate
        est = np.full((N, K), np.nan, dtype=float)
    else:
        est = np.full((N, K), np.nan, dtype=float)

    est_means_ranked = np.take_along_axis(est, order, axis=1)

    play_prob_per_run = play_counts.astype(float) / float(n_steps)
    play_prob_per_run_ranked = np.take_along_axis(play_prob_per_run, order, axis=1)

    optimal_action_share_per_run = optimal_action_count_per_run.astype(float) / float(n_steps)

    # final estimation MSE per run; if estimates are unavailable, return NaNs
    if np.all(np.isfinite(est)):
        final_estimation_mse_per_run = np.mean((est - gang.means) ** 2, axis=1)
    else:
        final_estimation_mse_per_run = np.full(N, np.nan, dtype=float)
        finite_mask = np.isfinite(est)
        row_has_any = np.any(finite_mask, axis=1)
        if np.any(row_has_any):
            tmp = np.full(N, np.nan, dtype=float)
            for i in np.where(row_has_any)[0]:
                mask = finite_mask[i]
                tmp[i] = float(np.mean((est[i, mask] - gang.means[i, mask]) ** 2))
            final_estimation_mse_per_run = tmp

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

        "optimal_action_mean": optimal_action_mean,
        "optimal_action_var": optimal_action_var,
        "cum_optimal_action_mean": cum_optimal_action_mean,
        "cum_optimal_action_var": cum_optimal_action_var,
        "optimal_action_count_per_run": optimal_action_count_per_run.copy(),
        "optimal_action_share_per_run": optimal_action_share_per_run,

        "final_estimation_mse_per_run": final_estimation_mse_per_run,
        "final_regret_per_run": cum_regret_per_run.copy(),
    }

# -----------------------------
# Tuning: successive 1/3-ing with reduced resources
# -----------------------------

def tune_parameters(
    spec: AlgoSpec,
    means: np.ndarray,
    n_steps: int,
    seed: int = 123,
) -> Tuple[Dict[str, Any], float]:
    if len(spec.grid) == 1:
        p = spec.grid[0]
        means_r = means[: min(500, len(means))]
        res = run_bulk_experiment(spec, p, means=means_r, n_steps=n_steps, seed=seed)
        score = float(res["cum_regret_mean"][-1])
        return p, score

    eta = 3
    n_rounds = 3
    candidates = list(spec.grid)

    steps_schedule = [n_steps, n_steps, n_steps]
    N_schedule = [100, 300, 1000]

    final_scored: List[Tuple[float, Dict[str, Any]]] = []

    for r in range(n_rounds):
        n_steps_r = steps_schedule[r]
        N_r = N_schedule[r]
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

        scored.sort(key=lambda x: x[0])

        if r == n_rounds - 1:
            final_scored = scored
            break

        keep = max(1, len(scored) // eta)
        candidates = [p for _, p in scored[:keep]]

    best_score, best_params = final_scored[0]

    print(
        f"Best params for {spec.name} at n={n_steps}: "
        f"{best_params} (tune mean regret = {best_score:.4f})"
    )

    return best_params, best_score

# -----------------------------
# Tasks
# -----------------------------

def plot_parameters():
    # experiment constants
    K = 5
    n_steps = 10_000
    N = 1_000

    # shared bandit instances
    np.random.seed(0)
    means = np.random.rand(N, K)

    base_dir = _ensure_dir(os.path.join(OUT_DIR, "ex3_plot_parameters"))
    print(f"Writing data to: {base_dir}")

    index = []

    for spec in _algo_specs():
        print(f"\r\033[KPlotting parameters for algorithm: {spec.name}", end="")

        # algorithm-specific folder
        algo_dir = os.path.join(base_dir, spec.name)
        os.makedirs(algo_dir, exist_ok=True)

        algo_index = []

        # infer parameter name from grid, if any
        param_name = None
        if len(spec.grid) > 0 and len(spec.grid[0]) > 0:
            param_name = next(iter(spec.grid[0].keys()))

        # no-parameter algorithms: run once
        if param_name is None or len(spec.plotgrid) == 0:
            params = {}
            res = run_bulk_experiment(spec, params, means=means, n_steps=n_steps, seed=2000)

            run_dir = os.path.join(algo_dir, "default")
            os.makedirs(run_dir, exist_ok=True)
            _save_run(run_dir, res)

            algo_index.append({
                "params": params,
                "dir": "default",
                "mean_final_regret": float(res["cum_regret_mean"][-1]),
            })

            print(f"Done {spec.name}: mean final regret = {float(res['cum_regret_mean'][-1]):.4f}")

        else:
            for i, par in enumerate(spec.plotgrid):
                params = {param_name: par}
                res = run_bulk_experiment(
                    spec,
                    params,
                    means=means,
                    n_steps=n_steps,
                    seed=2000 + i,
                )

                run_name = f"{param_name}_{par}"
                run_dir = os.path.join(algo_dir, run_name)
                os.makedirs(run_dir, exist_ok=True)
                _save_run(run_dir, res)

                algo_index.append({
                    "params": params,
                    "dir": run_name,
                    "mean_final_regret": float(res["cum_regret_mean"][-1]),
                })

            print(f"Done {spec.name}: saved {len(spec.plotgrid)} parameter runs")

        with open(os.path.join(algo_dir, "index.json"), "w", encoding="utf-8") as f:
            json.dump(algo_index, f, indent=2)

        index.append({
            "algorithm": spec.name,
            "dir": spec.name,
            "parameter": param_name,
            "runs": len(algo_index),
        })

    # save lightweight top-level index
    with open(os.path.join(base_dir, "index.json"), "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)

    return

def best_at_n():
    # experiment constants
    K = 5
    N = 1_000
    horizons = [2_000, 4_000, 6_000, 8_000, 10_000]

    # shared bandit instances
    np.random.seed(0)
    means = np.random.rand(N, K)

    base_dir = _ensure_dir(os.path.join(OUT_DIR, "ex4_best_at_n"))
    print(f"Writing data to: {base_dir}")

    top_index = []

    for spec in _algo_specs():
        print(f"\r\033[KFinding best-at-n for algorithm: {spec.name}", end="")

        algo_dir = os.path.join(base_dir, spec.name)
        os.makedirs(algo_dir, exist_ok=True)

        algo_index = []

        for i, n_steps in enumerate(horizons):
            best_params, tune_score = tune_parameters(
                spec,
                means=means,
                n_steps=n_steps,
                seed=1000 + 100 * i,
            )

            res = run_bulk_experiment(
                spec,
                best_params,
                means=means,
                n_steps=n_steps,
                seed=2000 + 100 * i,
            )

            run_dir = os.path.join(algo_dir, f"n_{n_steps}")
            os.makedirs(run_dir, exist_ok=True)
            _save_run(run_dir, res)

            mean_cum_regret = float(res["cum_regret_mean"][-1])
            mean_regret_per_pull = mean_cum_regret / float(n_steps)

            entry = {
                "n_steps": n_steps,
                "best_params": best_params,
                "tune_score": float(tune_score),
                "mean_cumulative_regret": float(res["cum_regret_mean"][-1]),
                "dir": f"n_{n_steps}",
            }
            algo_index.append(entry)

            print(
                f"\r\033[K{spec.name} @ n={n_steps}: "
                f"best={best_params}, regret={float(res['cum_regret_mean'][-1]):.4f}"
            )

        with open(os.path.join(algo_dir, "index.json"), "w", encoding="utf-8") as f:
            json.dump(algo_index, f, indent=2)

        top_index.append({
            "algorithm": spec.name,
            "dir": spec.name,
            "runs": len(algo_index),
        })

    with open(os.path.join(base_dir, "index.json"), "w", encoding="utf-8") as f:
        json.dump(top_index, f, indent=2)

    print("\nDone.")

def compare_at_10000():
    # experiment constants
    K = 5
    n_steps = 10_000
    N = 1_000

    # shared bandit instances
    np.random.seed(0)
    means = np.random.rand(N, K)

    base_dir = _ensure_dir(os.path.join(OUT_DIR, "ex2_all_algorithms"))
    print(f"Writing data to: {base_dir}")

    index = []

    for spec in _algo_specs():
        print(f"\r\033[KRunning algorithm: {spec.name}", end="")

        # parameter tuning
        best_params, _ = tune_parameters(spec, means=means, n_steps=n_steps, seed=1000)
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
    compare_at_10000()
    plot_parameters()
    best_at_n()
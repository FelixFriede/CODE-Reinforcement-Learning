# script/Task_1_8_performance.py
# This file is CORE.
# However, data saving (instead of straight image rendering) was introduced late and is purely AI generated and not proof read.

# general
import numpy as np
import sys, os, json
from typing import Any

# bandit & algorithm
from src.bandits import Gang_of_Bandits
from src.etc import ETCBulkAlgorithm

# global utilities
from util.io_helpers import OUT_DIR


# ----------------------
# local utilities
# ----------------------

def _mean_var(x, axis=0):
    # Return (mean, var) along axis using population variance.

    x = np.asarray(x)
    return x.mean(axis=axis), x.var(axis=axis)


def get_experiment_out_dir(result: dict[str, Any]) -> str:

    m = result["exploration_rounds"]
    exp_dir = os.path.join(OUT_DIR, "ex1_etc", f"etc_m{m}")
    os.makedirs(exp_dir, exist_ok=True)
    return exp_dir


# ----------------------
# data saving helpers (not proof read)
# ----------------------

def _is_np_array_like(x: Any) -> bool:
    return isinstance(x, np.ndarray)


def _to_jsonable(x: Any):
    # keep meta small + robust (no giant arrays here)
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    if isinstance(x, dict):
        return {str(k): _to_jsonable(v) for k, v in x.items()}
    # fallback: string repr for odd stuff (e.g. enums)
    return str(x)


def split_res(res: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    arrays: dict[str, np.ndarray] = {}
    meta: dict[str, Any] = {}
    for k, v in res.items():
        if _is_np_array_like(v):
            arrays[k] = v
        else:
            meta[k] = _to_jsonable(v)
    return arrays, meta


def save_run(exp_dir: str, res: dict[str, Any]) -> None:
    arrays, meta = split_res(res)

    # Arrays go to a compressed NPZ
    np.savez_compressed(os.path.join(exp_dir, "result_arrays.npz"), **arrays)

    # Meta goes to small JSON
    with open(os.path.join(exp_dir, "result_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # Convenience pointer for render script
    with open(os.path.join(exp_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "arrays": "result_arrays.npz",
                "meta": "result_meta.json",
            },
            f,
            indent=2,
        )


# ----------------------
# core
# ----------------------

def run_bulk_etc_experiment(
    exploration_rounds: int,
    n_steps: int = 10_000,
    N: int = 1_000,
    n_arms: int = 10,
    seed: int | None = None,
):
    """
    Run N independent ETC experiments in parallel (vectorized) for n_steps.

    Returns a dict with means + variances for:
      (a) regret over time (instant + cumulative)
      (b) correct-action rate over time
      (c) rank-aligned mean estimates vs true means over time
      (d) arm-choice probabilities over time
    """
    if seed is not None:
        np.random.seed(seed)

    gang = Gang_of_Bandits(n_bandits=N, n_arms=n_arms, distribution="gaussian")
    algo = ETCBulkAlgorithm(gang, exploration_rounds=exploration_rounds)

    # True best arm per run (row)
    true_best = np.argmax(gang.means, axis=1)
    rows = np.arange(N)

    # Rank alignment indices (best->worst) for each run
    order = np.argsort(gang.means, axis=1)[:, ::-1]  # (N, K)
    true_means_ranked = np.take_along_axis(gang.means, order, axis=1)  # (N, K)
    true_means_ranked_mean, true_means_ranked_var = _mean_var(true_means_ranked, axis=0)

    # Storage over time
    inst_regret_mean = np.empty(n_steps, dtype=float)
    inst_regret_var = np.empty(n_steps, dtype=float)

    cum_regret_mean = np.empty(n_steps, dtype=float)
    cum_regret_var = np.empty(n_steps, dtype=float)

    correct_rate_mean = np.empty(n_steps, dtype=float)
    correct_rate_var = np.empty(n_steps, dtype=float)

    # (d) probability of choosing each arm at time t
    arm_prob_mean = np.empty((n_steps, n_arms), dtype=float)
    arm_prob_var = np.empty((n_steps, n_arms), dtype=float)

    # (c) ranked empirical mean estimates over time (mean+var across runs)
    est_ranked_mean = np.empty((n_steps, n_arms), dtype=float)
    est_ranked_var = np.empty((n_steps, n_arms), dtype=float)

    # Track cumulative regret per run to compute mean/var each t
    cum_regret_per_run = np.zeros(N, dtype=float)

    for t in range(n_steps):
        if t % 1000 == 0:
            sys.stdout.write(f"\rRunning  ETC experiment | gang size N={N} | exploration rounds m={exploration_rounds} | step {t}/{n_steps}")
            sys.stdout.flush()

        chosen, rewards = algo.step()

        # (a) Instant regret computed from *true means* (more stable than reward-based regret)
        mu_star = gang.means[rows, true_best]
        mu_chosen = gang.means[rows, chosen]
        inst_regret = mu_star - mu_chosen  # (N,)

        cum_regret_per_run += inst_regret

        inst_regret_mean[t], inst_regret_var[t] = _mean_var(inst_regret, axis=0)
        cum_regret_mean[t], cum_regret_var[t] = _mean_var(cum_regret_per_run, axis=0)

        # (b) correct-action rate
        correct = (chosen == true_best).astype(float)
        correct_rate_mean[t], correct_rate_var[t] = _mean_var(correct, axis=0)

        # (d) probabilities of choosing each arm at time t across runs
        # bincount is fast for 10 arms
        counts = np.bincount(chosen, minlength=n_arms).astype(float)
        probs = counts / N  # shape (K,)
        arm_prob_mean[t] = probs
        # For variance per arm across runs, we treat each run as a one-hot draw:
        # var(1{A=a}) across runs = p*(1-p) (population variance).
        arm_prob_var[t] = probs * (1.0 - probs)

        # (c) ranked empirical estimates over time
        # estimates per run per arm
        denom = np.maximum(algo.arm_pull_counts, 1)
        est = algo.arm_reward_sums / denom  # (N, K)
        est_ranked = np.take_along_axis(est, order, axis=1)  # (N, K)

        est_ranked_mean[t], est_ranked_var[t] = _mean_var(est_ranked, axis=0)

    return {
        "exploration_rounds": exploration_rounds,
        "n_steps": n_steps,
        "N": N,
        "n_arms": n_arms,
        "true_means_ranked_mean": true_means_ranked_mean,
        "true_means_ranked_var": true_means_ranked_var,
        "inst_regret_mean": inst_regret_mean,
        "inst_regret_var": inst_regret_var,
        "cum_regret_mean": cum_regret_mean,
        "cum_regret_var": cum_regret_var,
        "correct_rate_mean": correct_rate_mean,
        "correct_rate_var": correct_rate_var,
        "arm_prob_mean": arm_prob_mean,
        "arm_prob_var": arm_prob_var,
        "est_ranked_mean": est_ranked_mean,
        "est_ranked_var": est_ranked_var,
    }


# ----------------------
# main
# ----------------------

def main():
    # ms = [5,10,15,20,25,30,35,40,45,50]
    ms = [12,13,14,15,16,17,18,19,20,21,22]

    sweep_dir = os.path.join(OUT_DIR, "ex1_etc")
    os.makedirs(sweep_dir, exist_ok=True)

    index = [] 

    for m in ms:
        res = run_bulk_etc_experiment(
            exploration_rounds=m,
            n_steps=10_000,
            N=1_000,
            n_arms=10,
        )

        exp_dir = get_experiment_out_dir(res)
        save_run(exp_dir, res)

        index.append({"m": int(m), "dir": os.path.relpath(exp_dir, sweep_dir)})

        print(
            f"\r\033[KSaved ETC experiment | N=1000 | m={m} | n_steps=10000 -> {exp_dir}"
        )

    with open(os.path.join(sweep_dir, "index.json"), "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)


if __name__ == "__main__":
    main()
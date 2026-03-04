import numpy as np
import matplotlib.pyplot as plt
import sys
import os

from src.bandits import Gang_of_Bandits
from src.etc import ETCBulkAlgorithm
from util.io_helpers import OUT_DIR


def _mean_var(x, axis=0):
    """Return (mean, var) along axis using population variance."""
    x = np.asarray(x)
    return x.mean(axis=axis), x.var(axis=axis)


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
        if t % 250 == 0:
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


# -----------------------------
# Plotting helpers
# -----------------------------

def get_experiment_out_dir(result):
    """
    Create and return an output directory for the experiment.
    """
    m = result["exploration_rounds"]
    exp_dir = os.path.join(OUT_DIR, f"etc_m{m}")
    os.makedirs(exp_dir, exist_ok=True)
    return exp_dir

def _ensure_out_dir():
    os.makedirs(OUT_DIR, exist_ok=True)


def plot_regret_over_time(result):
    exp_dir = get_experiment_out_dir(result)
    t = np.arange(result["n_steps"])

    plt.figure()
    plt.plot(t, result["cum_regret_mean"])
    plt.xlabel("t")
    plt.ylabel("Average cumulative regret")
    plt.title(f"ETC cumulative regret (m={result['exploration_rounds']})")

    plt.savefig(os.path.join(exp_dir, "regret_cumulative.png"), dpi=150)
    plt.close()

    plt.figure()
    plt.plot(t, result["inst_regret_mean"])
    plt.xlabel("t")
    plt.ylabel("Average instant regret")
    plt.title(f"ETC instant regret (m={result['exploration_rounds']})")

    plt.savefig(os.path.join(exp_dir, "regret_instant.png"), dpi=150)
    plt.close()

def plot_correct_action_rate_over_time(result):
    exp_dir = get_experiment_out_dir(result)
    t = np.arange(result["n_steps"])

    plt.figure()
    plt.plot(t, result["correct_rate_mean"])
    plt.xlabel("t")
    plt.ylabel("Correct action rate")
    plt.title(f"ETC correct action rate (m={result['exploration_rounds']})")
    plt.ylim(-0.05, 1.05)

    plt.savefig(os.path.join(exp_dir, "correct_action_rate.png"), dpi=150)
    plt.close()

def plot_ranked_estimates_vs_true_over_time(result, ranks_to_plot=(0, 1, 2, 9)):
    exp_dir = get_experiment_out_dir(result)

    t = np.arange(result["n_steps"])
    true_ranked = result["true_means_ranked_mean"]

    plt.figure()

    for r in ranks_to_plot:
        plt.plot(t, result["est_ranked_mean"][:, r], label=f"estimate rank {r+1}")
        plt.hlines(true_ranked[r], 0, result["n_steps"] - 1, linestyles="dashed")

    plt.xlabel("t")
    plt.ylabel("Mean estimate")
    plt.title(f"ETC ranked estimates vs true means (m={result['exploration_rounds']})")
    plt.legend()

    plt.savefig(os.path.join(exp_dir, "ranked_estimates.png"), dpi=150)
    plt.close()

def plot_arm_choice_probabilities_over_time(result):
    exp_dir = get_experiment_out_dir(result)
    t = np.arange(result["n_steps"])

    plt.figure()

    for a in range(result["n_arms"]):
        plt.plot(t, result["arm_prob_mean"][:, a], label=f"arm {a}")

    plt.xlabel("t")
    plt.ylabel("P(choose arm)")
    plt.title(f"ETC arm choice probabilities (m={result['exploration_rounds']})")
    plt.legend(ncol=2)
    plt.ylim(-0.05, 1.05)

    plt.savefig(os.path.join(exp_dir, "arm_probabilities.png"), dpi=150)
    plt.close()
    

def main():
    ms = [1,5,10,25,50,100,250]
    for m in ms:
        res = run_bulk_etc_experiment(exploration_rounds=m, n_steps=10_000, N=1_000, n_arms=10, seed=123)
        plot_regret_over_time(res)
        plot_correct_action_rate_over_time(res)
        plot_ranked_estimates_vs_true_over_time(res)
        plot_arm_choice_probabilities_over_time(res)
        print(f"\rFinished ETC experiment | gang size N={1_000} | exploration rounds m={m} | steps n={10_000}")

if __name__ == "__main__":
    main()
# src/experiment_etc.py
# IMPORTANT: This is a working version and not part of the final assigment, all features are legacy.

# This is the naive implementation if Blatt 1, Aufgabe 8.
# Pulling 1000*10000*5 arms one at a time seems insane given that vectorization exists.
import numpy as np
import matplotlib.pyplot as plt

from src.bandits import Bandit
from src.etc import ETCAlgorithm
from util.io_helpers import log, out


def run_single_experiment(n_arms, n_steps, exploration_rounds):
    # Runs one ETC experiment on a Gaussian bandit and logs all metrics.

    bandit = Bandit(n_arms=n_arms, distribution="gaussian")
    algorithm = ETCAlgorithm(bandit, exploration_rounds)

    true_means = np.array(bandit.means)
    optimal_arm = np.argmax(true_means)

    regrets = np.zeros(n_steps)
    correct_actions = np.zeros(n_steps)
    mean_estimates = np.zeros((n_steps, n_arms))
    arm_probabilities = np.zeros((n_steps, n_arms))

    pull_counts = np.zeros(n_arms)
    reward_sums = np.zeros(n_arms)

    for t in range(n_steps):
        chosen_arm, reward = algorithm.step()

        pull_counts[chosen_arm] += 1
        reward_sums[chosen_arm] += reward

        for arm in range(n_arms):
            if pull_counts[arm] > 0:
                mean_estimates[t, arm] = reward_sums[arm] / pull_counts[arm]
            else:
                mean_estimates[t, arm] = 0.0

        optimal_reward = true_means[optimal_arm]
        regrets[t] = optimal_reward - true_means[chosen_arm]

        if chosen_arm == optimal_arm:
            correct_actions[t] = 1

        arm_probabilities[t, chosen_arm] = 1

    return {
        "regret": regrets,
        "correct": correct_actions,
        "mean_estimates": mean_estimates,
        "arm_probs": arm_probabilities,
        "true_means": true_means
    }


def run_multiple_experiments(N, n_arms, n_steps, exploration_rounds):
    # Runs N experiments and aggregates mean and variance statistics.

    all_regrets = []
    all_correct = []
    all_means = []
    all_probs = []

    for i in range(N):
        print(f"\r{i + 1}/{N}", end="", flush=True)
        result = run_single_experiment(n_arms, n_steps, exploration_rounds)
        all_regrets.append(result["regret"])
        all_correct.append(result["correct"])
        all_means.append(result["mean_estimates"])
        all_probs.append(result["arm_probs"])

    all_regrets = np.array(all_regrets)
    all_correct = np.array(all_correct)
    all_means = np.array(all_means)
    all_probs = np.array(all_probs)

    aggregated = {
        "regret_mean": np.mean(all_regrets, axis=0),
        "regret_var": np.var(all_regrets, axis=0),
        "correct_mean": np.mean(all_correct, axis=0),
        "correct_var": np.var(all_correct, axis=0),
        "means_mean": np.mean(all_means, axis=0),
        "means_var": np.var(all_means, axis=0),
        "probs_mean": np.mean(all_probs, axis=0),
        "probs_var": np.var(all_probs, axis=0)
    }

    return aggregated


def plot_results(results, true_means, exploration_rounds):
    # Creates all required plots.

    T = len(results["regret_mean"])
    time = np.arange(T)

    plt.figure()
    plt.plot(time, np.cumsum(results["regret_mean"]))
    plt.title(f"Cumulative Regret (m={exploration_rounds})")
    plt.xlabel("Time")
    plt.ylabel("Regret")
    plt.show()

    plt.figure()
    plt.plot(time, results["correct_mean"])
    plt.title("Correct Action Rate")
    plt.xlabel("Time")
    plt.ylabel("Probability of Optimal Arm")
    plt.show()

    plt.figure()
    for arm in range(len(true_means)):
        plt.plot(time, results["means_mean"][:, arm], label=f"Arm {arm}")
        plt.hlines(true_means[arm], 0, T, linestyles="dashed")
    plt.title("Estimated Means vs True Means")
    plt.xlabel("Time")
    plt.ylabel("Mean Reward")
    plt.legend()
    plt.show()

    plt.figure()
    for arm in range(len(true_means)):
        plt.plot(time, results["probs_mean"][:, arm], label=f"Arm {arm}")
    plt.title("Arm Selection Probabilities")
    plt.xlabel("Time")
    plt.ylabel("Probability")
    plt.legend()
    plt.show()



def find_optimal_m(m_values, N, n_arms, n_steps):
    # Searches for m minimizing total regret.

    regrets = []

    for m in m_values:
        print(f"Exploration rounds m={m}. ")
        results = run_multiple_experiments(N, n_arms, n_steps, m)
        total_regret = np.sum(results["regret_mean"])
        regrets.append(total_regret)

    optimal_index = np.argmin(regrets)
    print("\n")
    return m_values[optimal_index], regrets


if __name__ == "__main__":
    n_arms = 10
    n_steps = 10000
    N = 1000

    # we really should solve this theoretically.
    m_values = [50, 100, 200, 500, 1000]

    optimal_m, regret_values = find_optimal_m(m_values, N, n_arms, n_steps)
    out(f"Optimal exploration rounds m = {optimal_m}", "results.txt")

    final_results = run_multiple_experiments(N, n_arms, n_steps, optimal_m)
    plot_results(final_results, np.zeros(n_arms), optimal_m)

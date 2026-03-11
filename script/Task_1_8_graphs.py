# script/Task_1_8_graphs.py
# This is purely AI generated.

# general
import numpy as np
import matplotlib.pyplot as plt
import os, json
from typing import Any


# global utility
from util.io_helpers import OUT_DIR

# -----------------------------
# local utility
# -----------------------------

def get_experiment_out_dir(result):
    # Create and return an output directory for the experiment.

    m = result["exploration_rounds"]
    exp_dir = os.path.join(OUT_DIR, "ex1_etc", f"etc_m{m}")
    os.makedirs(exp_dir, exist_ok=True)
    return exp_dir


# -----------------------------
# Plotting helpers
# some of these are legacy.
# -----------------------------

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

def plot_cumulative_regret_comparison(results):
    os.makedirs(OUT_DIR, exist_ok=True)

    plt.figure()

    for res in results:
        t = np.arange(res["n_steps"])
        m = res["exploration_rounds"]

        plt.plot(
            t,
            res["cum_regret_mean"],
            label=f"m={m}"
        )

    plt.xlabel("t")
    plt.ylabel("Average cumulative regret")
    plt.title("ETC cumulative regret comparison")
    plt.legend()

    file_path = os.path.join(OUT_DIR, "ex1_etc", "etc_regret_comparison.png")
    plt.savefig(file_path, dpi=150)
    plt.close()

def plot_correct_action_rate_comparison(results):
 
    os.makedirs(OUT_DIR, exist_ok=True)

    # sort results by exploration rounds descending
    results_sorted = sorted(results, key=lambda r: r["exploration_rounds"], reverse=True)

    plt.figure()

    for res in results_sorted:
        t = np.arange(res["n_steps"])
        m = res["exploration_rounds"]

        plt.plot(
            t,
            res["correct_rate_mean"],
            label=f"m={m}"
        )

    plt.xlabel("t")
    plt.ylabel("Correct action rate")
    plt.title("ETC correct action rate comparison")
    plt.legend()

    file_path = os.path.join(OUT_DIR, "ex1_etc", "etc_correct_action_rate_comparison.png")
    plt.savefig(file_path, dpi=150)
    plt.close()

def plot_instant_regret_comparison(results):
    
    os.makedirs(OUT_DIR, exist_ok=True)

    # sort results by exploration rounds descending
    results_sorted = sorted(results, key=lambda r: r["exploration_rounds"], reverse=True)

    plt.figure()

    for res in results_sorted:
        t = np.arange(res["n_steps"])
        m = res["exploration_rounds"]

        plt.plot(t, res["inst_regret_mean"], label=f"m={m}")

    plt.xlabel("t")
    plt.ylabel("Average instant regret")
    plt.title("ETC instant regret comparison")
    plt.legend()

    file_path = os.path.join(OUT_DIR, "ex1_etc", "etc_instant_regret_comparison.png")
    plt.savefig(file_path, dpi=150)
    plt.close()
    

# -----------------------------
# loading data helpers
# -----------------------------

def load_run(exp_dir: str) -> dict[str, Any]:
    # Load meta
    with open(os.path.join(exp_dir, "result_meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)

    # Load arrays
    arrays_path = os.path.join(exp_dir, "result_arrays.npz")
    with np.load(arrays_path, allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}

    return {**meta, **arrays}


# -----------------------------
# main
# -----------------------------


def main():
    sweep_dir = os.path.join(OUT_DIR, "ex1_etc")

    # Read index and load each run
    with open(os.path.join(sweep_dir, "index.json"), "r", encoding="utf-8") as f:
        index = json.load(f)

    results = []
    for entry in sorted(index, key=lambda d: d["m"]):
        exp_dir = os.path.join(sweep_dir, entry["dir"])
        res = load_run(exp_dir)
        results.append(res)

        m = res.get("exploration_rounds", entry["m"])
        print(f"\r\033[KPlotting ETC experiment | m={m}", end="")

        # Per-run plots (your functions should save to exp_dir internally,
        # or accept exp_dir; if they accept exp_dir, pass it here.)
        plot_ranked_estimates_vs_true_over_time(res)
        plot_arm_choice_probabilities_over_time(res)

        print(f"\r\033[KFinished plotting ETC experiment | m={m}")

    print(f"\r\033[KPlotting ETC comparisons across m", end="")
    plot_cumulative_regret_comparison(results)
    plot_instant_regret_comparison(results)
    plot_correct_action_rate_comparison(results)
    print(f"\r\033[KFinished plotting ETC comparisons across m")

if __name__ == "__main__":
    main()
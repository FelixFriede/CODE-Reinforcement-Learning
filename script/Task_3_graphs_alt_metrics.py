from __future__ import annotations
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Any, Dict, List

from util.io_helpers import OUT_DIR, _ensure_dir


# -----------------------------
# loading
# -----------------------------

def _load_run(exp_dir: str) -> Dict[str, Any]:
    with open(os.path.join(exp_dir, "result_meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)

    with np.load(os.path.join(exp_dir, "result_arrays.npz"), allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}

    return {**meta, **arrays}


def _load_selected_results(base_dir: str, selected_algos: List[str]) -> List[Dict[str, Any]]:
    index_path = os.path.join(base_dir, "index.json")
    with open(index_path, "r", encoding="utf-8") as f:
        index = json.load(f)

    index_by_algo = {entry["algorithm"]: entry for entry in index}

    results: List[Dict[str, Any]] = []
    for algo in selected_algos:
        entry = index_by_algo.get(algo)
        if entry is None:
            print(f"Warning: algorithm '{algo}' not found in {index_path}; skipping.")
            continue

        run_dir = os.path.join(base_dir, entry["dir"])
        if not os.path.isdir(run_dir):
            print(f"Warning: directory missing for algorithm '{algo}': {run_dir}; skipping.")
            continue

        results.append(_load_run(run_dir))

    return results


# -----------------------------
# labels / helpers
# -----------------------------

def _params_to_str(params: Dict[str, Any]) -> str:
    if not params:
        return ""
    return ", ".join(f"{k}={v}" for k, v in params.items())


def _result_label(res: Dict[str, Any], include_params: bool = True) -> str:
    name = res.get("name", "unknown")
    params = res.get("params", {})
    if include_params and params:
        return f"{name} ({_params_to_str(params)})"
    return name


def _ci95_from_var(var: np.ndarray, n_runs: int) -> np.ndarray:
    return 1.96 * np.sqrt(var / float(n_runs))


# -----------------------------
# plots for task c)
# -----------------------------

def plot_optimal_arm_probability_over_time(results: List[Dict[str, Any]], out_dir: str):
    """
    Probability of choosing the true best arm at each time t.
    """
    _ensure_dir(out_dir)

    n_steps = int(results[0]["n_steps"])
    n_runs = int(results[0]["N"])
    t = np.arange(n_steps)

    fig, ax = plt.subplots(figsize=(8.5, 5.2))

    for res in results:
        mean = res["optimal_action_mean"]
        var = res["optimal_action_var"]
        ci = _ci95_from_var(var, n_runs)

        label = _result_label(res, include_params=True)
        ax.plot(t, mean, label=label)
        ax.fill_between(t, mean - ci, mean + ci, alpha=0.15)

    ax.set_xlabel("Pull count")
    ax.set_ylabel("Probability of choosing the true best arm")
    ax.set_title("Optimal-arm selection probability over time")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0, fontsize=9)
    plt.subplots_adjust(right=0.72)
    plt.savefig(os.path.join(out_dir, "compare_optimal_arm_probability_ci.png"), dpi=180)
    plt.close()


def plot_cumulative_correct_actions_over_time(results: List[Dict[str, Any]], out_dir: str):
    """
    Cumulative number of correctly chosen actions up to time t.
    """
    _ensure_dir(out_dir)

    n_steps = int(results[0]["n_steps"])
    n_runs = int(results[0]["N"])
    t = np.arange(n_steps)

    fig, ax = plt.subplots(figsize=(8.5, 5.2))

    for res in results:
        mean = res["cum_optimal_action_mean"]
        var = res["cum_optimal_action_var"]
        ci = _ci95_from_var(var, n_runs)

        label = _result_label(res, include_params=True)
        ax.plot(t, mean, label=label)
        ax.fill_between(t, mean - ci, mean + ci, alpha=0.15)

    ax.set_xlabel("Pull count")
    ax.set_ylabel("Average cumulative number of correct actions")
    ax.set_title("Cumulative correctly chosen actions over time")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0, fontsize=9)
    plt.subplots_adjust(right=0.72)
    plt.savefig(os.path.join(out_dir, "compare_cumulative_correct_actions_ci.png"), dpi=180)
    plt.close()


def plot_final_best_arm_share_bar(results: List[Dict[str, Any]], out_dir: str):
    """
    End-of-horizon average share of pulls spent on the true best arm.
    """
    _ensure_dir(out_dir)

    labels = [_result_label(res, include_params=False) for res in results]
    values = [float(np.mean(res["optimal_action_share_per_run"])) for res in results]

    order = np.argsort(values)[::-1]
    labels = [labels[i] for i in order]
    values = [values[i] for i in order]

    plt.figure(figsize=(max(8, 0.75 * len(labels)), 5))
    plt.bar(np.arange(len(labels)), values)
    plt.xticks(np.arange(len(labels)), labels, rotation=45, ha="right")
    plt.ylabel("Average share of optimal actions")
    plt.title("Final optimal-arm selection share")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "compare_final_best_arm_share_bar.png"), dpi=180)
    plt.close()


def plot_final_estimation_mse_bar(results: List[Dict[str, Any]], out_dir: str):
    """
    Mean final estimation MSE of arm means.
    Lower is better.
    """
    _ensure_dir(out_dir)

    labels = []
    values = []

    for res in results:
        mse = np.asarray(res["final_estimation_mse_per_run"], dtype=float)
        mse = mse[np.isfinite(mse)]
        if len(mse) == 0:
            continue
        labels.append(_result_label(res, include_params=False))
        values.append(float(np.mean(mse)))

    if not values:
        print("Warning: no finite estimation MSE values found; skipping MSE bar plot.")
        return

    order = np.argsort(values)
    labels = [labels[i] for i in order]
    values = [values[i] for i in order]

    plt.figure(figsize=(max(8, 0.75 * len(labels)), 5))
    plt.bar(np.arange(len(labels)), values)
    plt.xticks(np.arange(len(labels)), labels, rotation=45, ha="right")
    plt.ylabel("Mean final estimation MSE")
    plt.title("Final arm-mean estimation error")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "compare_final_estimation_mse_bar.png"), dpi=180)
    plt.close()


def plot_final_estimation_mse_boxplot(results: List[Dict[str, Any]], out_dir: str):
    """
    Distribution of final estimation MSE across runs.
    """
    _ensure_dir(out_dir)

    data = []
    labels = []

    for res in results:
        mse = np.asarray(res["final_estimation_mse_per_run"], dtype=float)
        mse = mse[np.isfinite(mse)]
        if len(mse) == 0:
            continue
        data.append(mse)
        labels.append(_result_label(res, include_params=False))

    if not data:
        print("Warning: no finite estimation MSE values found; skipping MSE boxplot.")
        return

    plt.figure(figsize=(max(10, 0.6 * len(labels)), 5.5))
    plt.boxplot(data, showfliers=False)
    plt.xticks(np.arange(1, len(labels) + 1), labels, rotation=45, ha="right")
    plt.ylabel("Final estimation MSE")
    plt.title("Final arm-mean estimation error distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "compare_final_estimation_mse_boxplot.png"), dpi=180)
    plt.close()


def save_alternative_metrics_summary(results: List[Dict[str, Any]], out_dir: str):
    summary = []

    for res in results:
        mse = np.asarray(res["final_estimation_mse_per_run"], dtype=float)
        mse = mse[np.isfinite(mse)]

        summary.append({
            "algorithm": res.get("name", ""),
            "params": res.get("params", {}),
            "n_steps": int(res["n_steps"]),
            "N": int(res["N"]),
            "mean_optimal_arm_probability_final_share": float(np.mean(res["optimal_action_share_per_run"])),
            "mean_final_correct_actions": float(np.mean(res["optimal_action_count_per_run"])),
            "mean_final_estimation_mse": float(np.mean(mse)) if len(mse) > 0 else None,
        })

    with open(os.path.join(out_dir, "summary_alternative_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


# -----------------------------
# main
# -----------------------------

def main():
    base_dir = _ensure_dir(os.path.join(OUT_DIR, "ex2_all_algorithms_bernoulli"))
    print(f"Reading results from: {base_dir}")

    selected_algos = [
        "ETC",
        "EpsGrdy",
        "Eps0Grdy",
        "BernoulliUCB",   # or "UCB" if you renamed it
        "BltzSM",
        "PG",
    ]

    results = _load_selected_results(base_dir, selected_algos)

    if not results:
        raise RuntimeError("No results loaded. Check selected_algos and index.json.")

    plot_optimal_arm_probability_over_time(results, base_dir)
    plot_cumulative_correct_actions_over_time(results, base_dir)
    plot_final_best_arm_share_bar(results, base_dir)
    plot_final_estimation_mse_bar(results, base_dir)
    plot_final_estimation_mse_boxplot(results, base_dir)
    save_alternative_metrics_summary(results, base_dir)

    print("Done.")


if __name__ == "__main__":
    main()
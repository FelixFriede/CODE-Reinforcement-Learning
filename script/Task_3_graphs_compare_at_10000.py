# scripts/Task_3_graphs_compare_at_10000.py
# Cleaned-up comparison plots for ex2_all_algorithms_bernoulli at n = 10_000

from __future__ import annotations
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Any, Dict, List

from util.io_helpers import OUT_DIR, _ensure_dir


# -----------------------------
# local utilities
# -----------------------------

def _ci95_from_var(var: np.ndarray, N: int) -> np.ndarray:
    # 95% CI half-width for the mean using normal approximation
    return 1.96 * np.sqrt(var / float(N))


def _load_run(exp_dir: str) -> Dict[str, Any]:
    with open(os.path.join(exp_dir, "result_meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)

    with np.load(os.path.join(exp_dir, "result_arrays.npz"), allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}

    return {**meta, **arrays}


def _label_from_result(res: Dict[str, Any]) -> str:
    name = res.get("name", "")
    params = res.get("params", {})

    if not params:
        return name

    parts = [f"{k}={v}" for k, v in params.items()]
    return f"{name} ({', '.join(parts)})"


def _mean_final_regret(res: Dict[str, Any]) -> float:
    return float(res["cum_regret_mean"][-1])


# -----------------------------
# plotting
# -----------------------------

def plot_regret_with_ci(results: List[Dict[str, Any]], out_dir: str, kind: str = "cum"):
    """
    kind: "cum" or "inst"
    """
    if not results:
        return

    _ensure_dir(out_dir)

    n_steps = results[0]["n_steps"]
    N = results[0]["N"]
    t = np.arange(n_steps)

    fig, ax = plt.subplots(figsize=(8.5, 5.2))

    for r in results:
        if kind == "cum":
            mean = r["cum_regret_mean"]
            var = r["cum_regret_var"]
            ylabel = "Average cumulative regret"
            title = "Cumulative regret at n = 10,000"
            fname = "compare_cumulative_regret_ci.png"
        else:
            mean = r["inst_regret_mean"]
            var = r["inst_regret_var"]
            ylabel = "Average instant regret"
            title = "Instant regret at n = 10,000"
            fname = "compare_instant_regret_ci.png"

        ci = _ci95_from_var(var, N)
        label = _label_from_result(r)

        ax.plot(t, mean, label=label)
        ax.fill_between(t, mean - ci, mean + ci, alpha=0.15)

    ax.set_xlabel("Pull count t")
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        fontsize=9,
        ncol=1,
    )

    plt.subplots_adjust(right=0.72)
    plt.savefig(os.path.join(out_dir, fname), dpi=180, bbox_inches="tight")
    plt.close()


def plot_final_regret_boxplot(results: List[Dict[str, Any]], out_dir: str):
    """
    Boxplot of final cumulative regret per algorithm.
    """
    if not results:
        return

    _ensure_dir(out_dir)

    data = [r["final_regret_per_run"] for r in results]
    labels = [_label_from_result(r) for r in results]

    plt.figure(figsize=(max(10, 0.7 * len(labels)), 5.4))
    plt.boxplot(data, showfliers=False)
    plt.xticks(np.arange(1, len(labels) + 1), labels, rotation=35, ha="right")
    plt.ylabel("Cumulative regret at n = 10,000")
    plt.title("Final regret distributions by algorithm")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "compare_final_regret_boxplot.png"), dpi=180)
    plt.close()


def plot_mean_final_regret_bar(results: List[Dict[str, Any]], out_dir: str):
    """
    Bar chart of mean final cumulative regret.
    """
    if not results:
        return

    _ensure_dir(out_dir)

    labels = [_label_from_result(r) for r in results]
    values = [_mean_final_regret(r) for r in results]

    order = np.argsort(values)
    labels = [labels[i] for i in order]
    values = [values[i] for i in order]

    x = np.arange(len(labels))

    plt.figure(figsize=(max(10, 0.75 * len(labels)), 5.4))
    plt.bar(x, values)
    plt.xticks(x, labels, rotation=35, ha="right")
    plt.ylabel("Mean cumulative regret at n = 10,000")
    plt.title("Mean final regret by algorithm")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "compare_mean_final_regret_bar.png"), dpi=180)
    plt.close()


# -----------------------------
# main
# -----------------------------

def main():
    out_dir = _ensure_dir(os.path.join(OUT_DIR, "ex2_all_algorithms_bernoulli"))

    # Keep only algorithms you want in the comparison, in display order.
    selected_algos = [
        "ETC",
        "EpsGrdy",
        "Eps0Grdy",
        "UCB",
        "BltzSM",
        "PG",
    ]

    index_path = os.path.join(out_dir, "index.json")
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Missing index.json: {index_path}")

    with open(index_path, "r", encoding="utf-8") as f:
        index = json.load(f)

    index_by_algo = {entry["algorithm"]: entry for entry in index}

    results: List[Dict[str, Any]] = []
    for algo in selected_algos:
        entry = index_by_algo.get(algo)
        if entry is None:
            print(f"Warning: algorithm '{algo}' not found in index.json; skipping.")
            continue

        run_dir = os.path.join(out_dir, entry["dir"])
        if not os.path.exists(run_dir):
            print(f"Warning: missing directory for '{algo}': {run_dir}; skipping.")
            continue

        results.append(_load_run(run_dir))

    if not results:
        raise RuntimeError("No results loaded. Check selected_algos and saved experiment outputs.")

    print("Loaded algorithms:")
    for r in results:
        print(f"  - {_label_from_result(r)} | mean final regret = {_mean_final_regret(r):.4f}")

    plot_regret_with_ci(results, out_dir=out_dir, kind="cum")
    plot_regret_with_ci(results, out_dir=out_dir, kind="inst")
    plot_final_regret_boxplot(results, out_dir=out_dir)
    plot_mean_final_regret_bar(results, out_dir=out_dir)

    print(f"Saved plots to: {out_dir}")


if __name__ == "__main__":
    main()
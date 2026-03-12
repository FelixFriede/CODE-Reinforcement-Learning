# scripts/Task_3_plot_parameters_graphs.py
# Parameter-sweep plotting for ex3_plot_parameters

from __future__ import annotations
import os
import json
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


def _load_run(exp_dir: str) -> Dict[str, Any]:
    with open(os.path.join(exp_dir, "result_meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)

    arrays_path = os.path.join(exp_dir, "result_arrays.npz")
    with np.load(arrays_path, allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}

    return {**meta, **arrays}


def _label_from_result(res: Dict[str, Any], fallback: str = "") -> str:
    name = res.get("name", "")
    params = res.get("params", {})
    if not params:
        return name or fallback or "default"

    parts = [f"{k}={v}" for k, v in params.items()]
    return f"{name} ({', '.join(parts)})" if name else ", ".join(parts)


def _param_sort_key(res: Dict[str, Any]):
    params = res.get("params", {})
    if not params:
        return (-1, "")
    k = next(iter(params.keys()))
    v = params[k]
    try:
        return (0, float(v))
    except Exception:
        return (0, str(v))


# -----------------------------
# plotting
# -----------------------------

def plot_regret_with_ci(results: List[Dict[str, Any]], out_dir: str, kind: str = "cum"):
    """
    Overlay regret curves for different parameter choices of one algorithm.
    kind: "cum" or "inst"
    """
    if not results:
        return

    _ensure_dir(out_dir)
    results = sorted(results, key=_param_sort_key)

    n_steps = results[0]["n_steps"]
    N = results[0]["N"]
    t = np.arange(n_steps)

    algo_name = results[0].get("name", "algorithm")

    fig, ax = plt.subplots(figsize=(8, 5))

    for r in results:
        if kind == "cum":
            mean = r["cum_regret_mean"]
            var = r["cum_regret_var"]
            ylabel = "Average cumulative regret"
            title = f"{algo_name}: cumulative regret with 95% CI"
            fname = "regret_cumulative_ci.png"
        else:
            mean = r["inst_regret_mean"]
            var = r["inst_regret_var"]
            ylabel = "Average instant regret"
            title = f"{algo_name}: instant regret with 95% CI"
            fname = "regret_instant_ci.png"

        ci = _ci95_from_var(var, N)
        label = _label_from_result(r)

        ax.plot(t, mean, label=label)
        ax.fill_between(t, mean - ci, mean + ci, alpha=0.15)

    ax.set_xlabel("t")
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        borderaxespad=0,
        ncol=1,
        fontsize=9
    )

    plt.subplots_adjust(right=0.72)
    plt.savefig(os.path.join(out_dir, fname), dpi=160)
    plt.close()


def plot_final_regret_boxplot(results: List[Dict[str, Any]], out_dir: str):
    """
    Boxplot of final cumulative regrets across parameter choices of one algorithm.
    """
    if not results:
        return

    _ensure_dir(out_dir)
    results = sorted(results, key=_param_sort_key)

    algo_name = results[0].get("name", "algorithm")
    data = [r["final_regret_per_run"] for r in results]
    labels = [_label_from_result(r, fallback="default") for r in results]

    plt.figure(figsize=(max(10, 0.5 * len(labels)), 5))
    plt.boxplot(data, showfliers=False)
    plt.xticks(np.arange(1, len(labels) + 1), labels, rotation=90)
    plt.ylabel("Cumulative regret at n")
    plt.title(f"{algo_name}: final regret distributions by parameter")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "box_final_regrets.png"), dpi=160)
    plt.close()


def plot_final_regret_vs_parameter(results: List[Dict[str, Any]], out_dir: str):
    """
    Line plot of mean final cumulative regret against parameter value.
    Only produced for algorithms with exactly one varying parameter.
    """
    if not results:
        return

    _ensure_dir(out_dir)
    results = sorted(results, key=_param_sort_key)

    first_params = results[0].get("params", {})
    if len(first_params) != 1:
        return

    param_name = next(iter(first_params.keys()))
    algo_name = results[0].get("name", "algorithm")

    xs = []
    ys = []

    for r in results:
        params = r.get("params", {})
        if len(params) != 1 or param_name not in params:
            return
        try:
            xs.append(float(params[param_name]))
        except Exception:
            return
        ys.append(float(r["cum_regret_mean"][-1]))

    plt.figure(figsize=(7, 4.5))
    plt.plot(xs, ys, marker="o")
    plt.xlabel(param_name)
    plt.ylabel("Mean final cumulative regret")
    plt.title(f"{algo_name}: mean final regret vs {param_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "final_regret_vs_parameter.png"), dpi=160)
    plt.close()


# -----------------------------
# main
# -----------------------------

def main():
    base_dir = _ensure_dir(os.path.join(OUT_DIR, "ex3_plot_parameters"))
    print(f"Reading parameter sweep data from: {base_dir}")

    index_path = os.path.join(base_dir, "index.json")
    with open(index_path, "r", encoding="utf-8") as f:
        top_index = json.load(f)

    for algo_entry in top_index:
        algo_name = algo_entry["algorithm"]
        algo_dir = os.path.join(base_dir, algo_entry["dir"])
        algo_index_path = os.path.join(algo_dir, "index.json")

        if not os.path.exists(algo_index_path):
            print(f"Warning: missing {algo_index_path}; skipping {algo_name}.")
            continue

        with open(algo_index_path, "r", encoding="utf-8") as f:
            algo_index = json.load(f)

        results: List[Dict[str, Any]] = []
        for run_entry in algo_index:
            run_dir = os.path.join(algo_dir, run_entry["dir"])
            if not os.path.exists(run_dir):
                print(f"Warning: missing run dir {run_dir}; skipping.")
                continue
            results.append(_load_run(run_dir))

        if not results:
            print(f"Warning: no runs loaded for {algo_name}; skipping.")
            continue

        print(f"Plotting parameter comparison graphs for {algo_name}")

        plot_regret_with_ci(results, out_dir=algo_dir, kind="cum")
        plot_regret_with_ci(results, out_dir=algo_dir, kind="inst")
        plot_final_regret_boxplot(results, out_dir=algo_dir)
        plot_final_regret_vs_parameter(results, out_dir=algo_dir)


if __name__ == "__main__":
    main()
from __future__ import annotations
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Any, Dict, List

from util.io_helpers import OUT_DIR, _ensure_dir


def _load_run(exp_dir: str) -> Dict[str, Any]:
    with open(os.path.join(exp_dir, "result_meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)

    arrays_path = os.path.join(exp_dir, "result_arrays.npz")
    with np.load(arrays_path, allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}

    return {**meta, **arrays}


def _extract_single_param(params: Dict[str, Any]):
    if not params:
        return None, None
    if len(params) != 1:
        return None, None
    k = next(iter(params.keys()))
    return k, params[k]


def plot_best_parameter_vs_n(algo_name: str, algo_index: List[Dict[str, Any]], out_dir: str):
    if not algo_index:
        return

    algo_index = sorted(algo_index, key=lambda x: x["n_steps"])

    # Find parameter name
    param_name = None
    xs = []
    ys = []

    for entry in algo_index:
        k, v = _extract_single_param(entry.get("best_params", {}))
        if k is None:
            continue
        if param_name is None:
            param_name = k
        elif param_name != k:
            return  # mixed parameter types, skip
        xs.append(entry["n_steps"])
        ys.append(float(v))

    if param_name is None or not xs:
        return

    plt.figure(figsize=(6.5, 4.5))
    plt.plot(xs, ys, marker="o")
    plt.xticks(xs)
    plt.xlabel("Pull count n")
    plt.ylabel(f"Best {param_name}")
    plt.title(f"{algo_name}: ideal parameter vs pull count")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "best_parameter_vs_n.png"), dpi=160)
    plt.close()


def plot_best_cumulative_regret_vs_n(algo_name: str, algo_index: List[Dict[str, Any]], out_dir: str):
    if not algo_index:
        return

    algo_index = sorted(algo_index, key=lambda x: x["n_steps"])
    xs = [entry["n_steps"] for entry in algo_index]

    ys = []
    for entry in algo_index:
        if "mean_cumulative_regret" in entry:
            ys.append(float(entry["mean_cumulative_regret"]))
        else:
            # backward-compatible fallback for older saved data
            ys.append(float(entry["mean_final_regret"]))

    plt.figure(figsize=(6.5, 4.5))
    plt.plot(xs, ys, marker="o")
    plt.xticks(xs)
    plt.xlabel("Pull count n")
    plt.ylabel("Mean cumulative regret at n")
    plt.title(f"{algo_name}: tuned cumulative regret vs pull count")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "best_cumulative_regret_vs_n.png"), dpi=160)
    plt.close()

def plot_cumulative_regret_trajectories_for_best_at_n(
    algo_name: str,
    results: List[Dict[str, Any]],
    out_dir: str,
):
    """
    Plot cumulative regret over time for the tuned-best parameter at each horizon.
    Each curve stops at its own horizon.
    """
    if not results:
        return

    # Sort by horizon
    results = sorted(results, key=lambda r: int(r["n_steps"]))

    plt.figure(figsize=(8, 5))

    for res in results:
        n_steps = int(res["n_steps"])
        t = np.arange(n_steps)
        y = res["cum_regret_mean"]

        params = res.get("params", {})
        if params and len(params) == 1:
            k = next(iter(params.keys()))
            v = params[k]
            label = f"n={n_steps}, {k}={v}"
        else:
            label = f"n={n_steps}"

        plt.plot(t, y, label=label)

    plt.xlabel("t")
    plt.ylabel("Average cumulative regret")
    plt.title(f"{algo_name} cumulative regret comparison")
    plt.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "best_at_n_cumulative_regret_comparison.png"), dpi=160)
    plt.close()


def main():
    base_dir = _ensure_dir(os.path.join(OUT_DIR, "ex4_best_at_n"))
    print(f"Reading data from: {base_dir}")

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

        # load full saved runs
        results: List[Dict[str, Any]] = []
        for entry in algo_index:
            run_dir = os.path.join(algo_dir, entry["dir"])
            if not os.path.exists(run_dir):
                print(f"Warning: missing run dir {run_dir}; skipping.")
                continue
            results.append(_load_run(run_dir))

        print(f"Plotting best-at-n graphs for {algo_name}")

        plot_best_parameter_vs_n(algo_name, algo_index, algo_dir)
        plot_best_cumulative_regret_vs_n(algo_name, algo_index, algo_dir)
        plot_cumulative_regret_trajectories_for_best_at_n(algo_name, results, algo_dir)


if __name__ == "__main__":
    main()
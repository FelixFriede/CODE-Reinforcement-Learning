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
# labels
# -----------------------------

def _params_to_str(params: Dict[str, Any]) -> str:
    if not params:
        return ""
    return ", ".join(f"{k}={v}" for k, v in params.items())


def _result_label(res: Dict[str, Any], include_params: bool = False) -> str:
    name = res.get("name", "unknown")
    params = res.get("params", {})
    if include_params and params:
        return f"{name} ({_params_to_str(params)})"
    return name


# -----------------------------
# metric
# -----------------------------

def _best_arm_estimation_sq_error_per_run(res: Dict[str, Any]) -> np.ndarray:
    """
    Squared estimation error for the true best arm only.

    Because the saved arrays are rank-aligned, column 0 corresponds to the
    true best arm for every run:
      - true_means_ranked[:, 0]
      - est_means_ranked_final[:, 0]
    """
    true_best_mean = np.asarray(res["true_means_ranked"][:, 0], dtype=float)
    est_best_mean = np.asarray(res["est_means_ranked_final"][:, 0], dtype=float)

    sq_err = (est_best_mean - true_best_mean) ** 2
    return sq_err[np.isfinite(sq_err)]


# -----------------------------
# plotting
# -----------------------------

def plot_best_arm_estimation_error_boxplot(results: List[Dict[str, Any]], out_dir: str):
    """
    One boxplot comparing all algorithms on best-arm estimation squared error.
    """
    _ensure_dir(out_dir)

    data = []
    labels = []

    for res in results:
        sq_err = _best_arm_estimation_sq_error_per_run(res)
        if len(sq_err) == 0:
            print(f"Warning: no finite best-arm estimation errors for {_result_label(res)}; skipping.")
            continue

        data.append(sq_err)
        labels.append(_result_label(res, include_params=False))

    if not data:
        raise RuntimeError("No finite best-arm estimation error data found.")

    plt.figure(figsize=(max(10, 0.7 * len(labels)), 5.5))
    plt.boxplot(data, showfliers=False)
    plt.xticks(np.arange(1, len(labels) + 1), labels, rotation=45, ha="right")
    plt.ylabel("Squared error on true best arm estimate")
    plt.title("Final best-arm estimation error distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "compare_best_arm_estimation_error_boxplot.png"), dpi=180)
    plt.close()


def save_best_arm_estimation_summary(results: List[Dict[str, Any]], out_dir: str):
    summary = []

    for res in results:
        sq_err = _best_arm_estimation_sq_error_per_run(res)
        if len(sq_err) == 0:
            mean_sq_err = None
        else:
            mean_sq_err = float(np.mean(sq_err))

        summary.append({
            "algorithm": res.get("name", ""),
            "params": res.get("params", {}),
            "n_steps": int(res["n_steps"]),
            "N": int(res["N"]),
            "mean_best_arm_squared_error": mean_sq_err,
        })

    summary.sort(key=lambda x: float("inf") if x["mean_best_arm_squared_error"] is None else x["mean_best_arm_squared_error"])

    with open(os.path.join(out_dir, "summary_best_arm_estimation_error.json"), "w", encoding="utf-8") as f:
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
        "UCB",       # change to "BernoulliUCB" if that is your folder name
        "BltzSM",
        "PG",
    ]

    results = _load_selected_results(base_dir, selected_algos)

    if not results:
        raise RuntimeError("No results loaded. Check selected_algos and index.json.")

    plot_best_arm_estimation_error_boxplot(results, base_dir)
    save_best_arm_estimation_summary(results, base_dir)

    print("Done.")


if __name__ == "__main__":
    main()
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

        res = _load_run(run_dir)

        # nicer label for the equal-pulls baseline
        if algo == "ETC_equal":
            res["display_name"] = "Equal pulls (ETC, m=2000)"
        else:
            res["display_name"] = res.get("name", algo)

        results.append(res)

    return results


def _best_arm_sq_error_per_run(res: Dict[str, Any]) -> np.ndarray:
    """
    Squared estimation error for the true best arm only.
    Rank 0 corresponds to the true best arm.
    """
    true_best = np.asarray(res["true_means_ranked"][:, 0], dtype=float)
    est_best = np.asarray(res["est_means_ranked_final"][:, 0], dtype=float)

    sq_err = (est_best - true_best) ** 2
    return sq_err[np.isfinite(sq_err)]


def _best_arm_abs_error_per_run(res: Dict[str, Any]) -> np.ndarray:
    true_best = np.asarray(res["true_means_ranked"][:, 0], dtype=float)
    est_best = np.asarray(res["est_means_ranked_final"][:, 0], dtype=float)

    abs_err = np.abs(est_best - true_best)
    return abs_err[np.isfinite(abs_err)]


def plot_best_arm_sq_error_boxplot(results: List[Dict[str, Any]], out_dir: str):
    data = []
    labels = []

    for res in results:
        sq_err = _best_arm_sq_error_per_run(res)
        if len(sq_err) == 0:
            continue
        data.append(sq_err)
        labels.append(res.get("display_name", res.get("name", "unknown")))

    if not data:
        raise RuntimeError("No finite best-arm squared-error data found.")

    plt.figure(figsize=(max(10, 0.8 * len(labels)), 5.5))
    plt.boxplot(data, showfliers=False)
    plt.xticks(np.arange(1, len(labels) + 1), labels, rotation=45, ha="right")
    plt.ylabel("Squared error on best-arm mean estimate")
    plt.title("Final best-arm mean estimation error")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "compare_best_arm_estimation_sq_error_with_equal_pulls.png"), dpi=180)
    plt.close()


def plot_best_arm_abs_error_boxplot(results: List[Dict[str, Any]], out_dir: str):
    data = []
    labels = []

    for res in results:
        abs_err = _best_arm_abs_error_per_run(res)
        if len(abs_err) == 0:
            continue
        data.append(abs_err)
        labels.append(res.get("display_name", res.get("name", "unknown")))

    if not data:
        raise RuntimeError("No finite best-arm absolute-error data found.")

    plt.figure(figsize=(max(10, 0.8 * len(labels)), 5.5))
    plt.boxplot(data, showfliers=False)
    plt.xticks(np.arange(1, len(labels) + 1), labels, rotation=45, ha="right")
    plt.ylabel("Absolute error on best-arm mean estimate")
    plt.title("Final best-arm mean estimation absolute error")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "compare_best_arm_estimation_abs_error_with_equal_pulls.png"), dpi=180)
    plt.close()


def save_summary(results: List[Dict[str, Any]], out_dir: str):
    summary = []

    for res in results:
        sq_err = _best_arm_sq_error_per_run(res)
        abs_err = _best_arm_abs_error_per_run(res)

        summary.append({
            "algorithm": res.get("display_name", res.get("name", "")),
            "params": res.get("params", {}),
            "mean_best_arm_sq_error": float(np.mean(sq_err)) if len(sq_err) > 0 else None,
            "mean_best_arm_abs_error": float(np.mean(abs_err)) if len(abs_err) > 0 else None,
        })

    summary.sort(key=lambda x: float("inf") if x["mean_best_arm_sq_error"] is None else x["mean_best_arm_sq_error"])

    with open(os.path.join(out_dir, "summary_best_arm_estimation_with_equal_pulls.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def main():
    base_dir = _ensure_dir(os.path.join(OUT_DIR, "ex2_all_algorithms_bernoulli"))
    print(f"Reading results from: {base_dir}")

    selected_algos = [
        "ETC",
        "ETC_equal",
        "EpsGrdy",
        "Eps0Grdy",
        "UCB",      # or "BernoulliUCB" if that is your saved name
        "BltzSM",
        "PG",
    ]

    results = _load_selected_results(base_dir, selected_algos)

    if not results:
        raise RuntimeError("No results loaded. Check selected_algos and index.json.")

    plot_best_arm_sq_error_boxplot(results, base_dir)
    plot_best_arm_abs_error_boxplot(results, base_dir)
    save_summary(results, base_dir)

    print("Done.")


if __name__ == "__main__":
    main()
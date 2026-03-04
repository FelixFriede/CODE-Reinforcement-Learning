# tests/test_ucb_bulk.py

import numpy as np

from src.bandits import Gang_of_Bandits
from src.ucb import UCBulkAlgorithm, UCBSubGaussianBulkAlgorithm
from util.io_helpers import log, out


LOG_FILE = "ucb_bulk_test.log"


def summarize_array(arr, n_preview=8):
    arr = np.asarray(arr)
    if arr.size == 0:
        return f"shape={arr.shape}, empty"
    preview = arr.ravel()[:n_preview]
    try:
        mean = float(arr.mean())
        std = float(arr.std())
        mn = float(arr.min())
        mx = float(arr.max())
        return (
            f"shape={arr.shape}, mean={mean:.6f}, std={std:.6f}, "
            f"min={mn:.6f}, max={mx:.6f}, preview={preview.tolist()}"
        )
    except Exception:
        return f"shape={arr.shape}, preview={preview.tolist()}"


class TestInitShapesUCB:
    def run(self):
        log("Running TestInitShapesUCB", LOG_FILE)
        np.random.seed(0)

        gang = Gang_of_Bandits(n_bandits=1000, n_arms=7, distribution="gaussian")
        algo = UCBulkAlgorithm(gang, delta=0.1)

        assert algo.arm_pull_counts.shape == (1000, 7)
        assert algo.arm_value_estimates.shape == (1000, 7)
        assert algo.total_steps == 0

        log("TestInitShapesUCB produced data:", LOG_FILE)
        log("means summary: " + summarize_array(gang.means), LOG_FILE)
        log("arm_pull_counts summary: " + summarize_array(algo.arm_pull_counts), LOG_FILE)
        log("arm_value_estimates summary: " + summarize_array(algo.arm_value_estimates), LOG_FILE)
        log("", LOG_FILE)


class TestInitShapesUCBSubGaussian:
    def run(self):
        log("Running TestInitShapesUCBSubGaussian", LOG_FILE)
        np.random.seed(0)

        gang = Gang_of_Bandits(n_bandits=1000, n_arms=7, distribution="gaussian")
        algo = UCBSubGaussianBulkAlgorithm(gang, delta=0.1, sigma=1.0)

        assert algo.arm_pull_counts.shape == (1000, 7)
        assert algo.arm_value_estimates.shape == (1000, 7)
        assert algo.total_steps == 0

        log("TestInitShapesUCBSubGaussian produced data:", LOG_FILE)
        log("means summary: " + summarize_array(gang.means), LOG_FILE)
        log("arm_pull_counts summary: " + summarize_array(algo.arm_pull_counts), LOG_FILE)
        log("arm_value_estimates summary: " + summarize_array(algo.arm_value_estimates), LOG_FILE)
        log("", LOG_FILE)


class TestFirstKStepsTryAllArmsUCB:
    def run(self):
        log("Running TestFirstKStepsTryAllArmsUCB", LOG_FILE)
        np.random.seed(1)

        n_bandits = 2000
        n_arms = 6

        gang = Gang_of_Bandits(n_bandits=n_bandits, n_arms=n_arms, distribution="gaussian")
        algo = UCBulkAlgorithm(gang, delta=0.2)

        chosen_history = []
        for _ in range(n_arms):
            chosen, rewards = algo.step()
            assert chosen.shape == (n_bandits,)
            assert rewards.shape == (n_bandits,)
            chosen_history.append(chosen)

        # With +inf for unpulled arms and argmax tie-breaking, should pick 0,1,2,...,K-1
        for t, chosen in enumerate(chosen_history):
            assert np.all(chosen == t), f"At t={t}, expected all chosen=={t}"

        # After K steps, every arm should be pulled exactly once per bandit
        counts = algo.arm_pull_counts
        assert np.all(counts.sum(axis=1) == n_arms)
        assert np.all(counts == 1)

        log("TestFirstKStepsTryAllArmsUCB produced data:", LOG_FILE)
        log("chosen_first_step preview: " + str(chosen_history[0][:15].tolist()), LOG_FILE)
        log("counts row0: " + str(counts[0].tolist()), LOG_FILE)
        log("", LOG_FILE)


class TestFirstKStepsTryAllArmsUCBSubGaussian:
    def run(self):
        log("Running TestFirstKStepsTryAllArmsUCBSubGaussian", LOG_FILE)
        np.random.seed(1)

        n_bandits = 2000
        n_arms = 6

        gang = Gang_of_Bandits(n_bandits=n_bandits, n_arms=n_arms, distribution="gaussian")
        algo = UCBSubGaussianBulkAlgorithm(gang, delta=0.2, sigma=2.0)

        chosen_history = []
        for _ in range(n_arms):
            chosen, rewards = algo.step()
            assert chosen.shape == (n_bandits,)
            assert rewards.shape == (n_bandits,)
            chosen_history.append(chosen)

        for t, chosen in enumerate(chosen_history):
            assert np.all(chosen == t), f"At t={t}, expected all chosen=={t}"

        counts = algo.arm_pull_counts
        assert np.all(counts.sum(axis=1) == n_arms)
        assert np.all(counts == 1)

        log("TestFirstKStepsTryAllArmsUCBSubGaussian produced data:", LOG_FILE)
        log("counts row0: " + str(counts[0].tolist()), LOG_FILE)
        log("", LOG_FILE)


class TestStepIncrementsAndFiniteEstimates:
    def run(self):
        log("Running TestStepIncrementsAndFiniteEstimates", LOG_FILE)
        np.random.seed(2)

        n_bandits = 3000
        n_arms = 5
        steps = 50

        gang = Gang_of_Bandits(n_bandits=n_bandits, n_arms=n_arms, distribution="bernoulli")
        algo = UCBulkAlgorithm(gang, delta=0.1)

        for _ in range(steps):
            chosen, rewards = algo.step()
            assert chosen.shape == (n_bandits,)
            assert rewards.shape == (n_bandits,)

        assert algo.total_steps == steps
        assert np.all(algo.arm_pull_counts.sum(axis=1) == steps)

        # Estimates should be finite numbers
        assert np.all(np.isfinite(algo.arm_value_estimates))

        log("TestStepIncrementsAndFiniteEstimates produced data:", LOG_FILE)
        log("counts summary: " + summarize_array(algo.arm_pull_counts), LOG_FILE)
        log("estimates summary: " + summarize_array(algo.arm_value_estimates), LOG_FILE)
        log("", LOG_FILE)


class TestStatisticalSanityBestArmDominatesUCB:
    def run(self):
        log("Running TestStatisticalSanityBestArmDominatesUCB", LOG_FILE)
        np.random.seed(3)

        # Gang_of_Bandits sorts means descending => true best arm index is 0
        n_bandits = 10000
        n_arms = 10
        steps = 250

        gang = Gang_of_Bandits(n_bandits=n_bandits, n_arms=n_arms, distribution="gaussian")
        algo = UCBulkAlgorithm(gang, delta=0.1)

        for _ in range(steps):
            algo.step()

        learned_best = np.argmax(algo.arm_value_estimates, axis=1)
        hit_rate = float((learned_best == 0).mean())

        # Loose threshold to avoid flakiness; should be well above random baseline.
        assert hit_rate > 0.30

        log("TestStatisticalSanityBestArmDominatesUCB produced data:", LOG_FILE)
        log("learned_best summary: " + summarize_array(learned_best), LOG_FILE)
        log(f"hit_rate={hit_rate:.4f} (random baseline={1.0/n_arms:.4f})", LOG_FILE)
        log("", LOG_FILE)


class TestStatisticalSanityBestArmDominatesUCBSubGaussian:
    def run(self):
        log("Running TestStatisticalSanityBestArmDominatesUCBSubGaussian", LOG_FILE)
        np.random.seed(3)

        n_bandits = 10000
        n_arms = 10
        steps = 250

        gang = Gang_of_Bandits(n_bandits=n_bandits, n_arms=n_arms, distribution="gaussian")
        algo = UCBSubGaussianBulkAlgorithm(gang, delta=0.1, sigma=1.0)

        for _ in range(steps):
            algo.step()

        learned_best = np.argmax(algo.arm_value_estimates, axis=1)
        hit_rate = float((learned_best == 0).mean())

        assert hit_rate > 0.30

        log("TestStatisticalSanityBestArmDominatesUCBSubGaussian produced data:", LOG_FILE)
        log("learned_best summary: " + summarize_array(learned_best), LOG_FILE)
        log(f"hit_rate={hit_rate:.4f} (random baseline={1.0/n_arms:.4f})", LOG_FILE)
        log("", LOG_FILE)


def run_all_tests():
    tests = [
        TestInitShapesUCB(),
        TestInitShapesUCBSubGaussian(),
        TestFirstKStepsTryAllArmsUCB(),
        TestFirstKStepsTryAllArmsUCBSubGaussian(),
        TestStepIncrementsAndFiniteEstimates(),
        TestStatisticalSanityBestArmDominatesUCB(),
        TestStatisticalSanityBestArmDominatesUCBSubGaussian(),
    ]

    for test in tests:
        name = test.__class__.__name__
        try:
            test.run()
            log(f"{name}: PASS", LOG_FILE)
        except Exception as e:
            log(f"{name}: FAIL -> {type(e).__name__}: {e}", LOG_FILE)
            raise

    log("All ucb bulk algorithm tests completed successfully.", LOG_FILE)


if __name__ == "__main__":
    run_all_tests()
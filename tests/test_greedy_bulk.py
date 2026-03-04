# tests.test_greedy_bulk.py
# Equivalent in style/structure to etc_bulk_test.py :contentReference[oaicite:0]{index=0}

import numpy as np

from src.bandits import Gang_of_Bandits
from src.greedy import (
    GreedyBulkAlgorithm,
    EpsilonGreedyFixedBulkAlgorithm,
    EpsilonGreedyDecreasingBulkAlgorithm,
)
from util.io_helpers import log, out


LOG_FILE = "greedy_bulk_test.log"


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


class TestInitShapesGreedy:
    def run(self):
        log("Running TestInitShapesGreedy", LOG_FILE)
        np.random.seed(0)

        gang = Gang_of_Bandits(n_bandits=1000, n_arms=7, distribution="gaussian")
        algo = GreedyBulkAlgorithm(gang)

        assert algo.arm_pull_counts.shape == (1000, 7)
        assert algo.arm_value_estimates.shape == (1000, 7)
        assert algo.total_steps == 0

        log("TestInitShapesGreedy produced data:", LOG_FILE)
        log("means summary: " + summarize_array(gang.means), LOG_FILE)
        log("arm_pull_counts summary: " + summarize_array(algo.arm_pull_counts), LOG_FILE)
        log("arm_value_estimates summary: " + summarize_array(algo.arm_value_estimates), LOG_FILE)
        log("", LOG_FILE)


class TestGreedyAlwaysArm0Initially:
    def run(self):
        log("Running TestGreedyAlwaysArm0Initially", LOG_FILE)
        np.random.seed(1)

        n_bandits = 2000
        n_arms = 5

        gang = Gang_of_Bandits(n_bandits=n_bandits, n_arms=n_arms, distribution="gaussian")
        algo = GreedyBulkAlgorithm(gang)

        # All Q_hat start equal (0), argmax tie-break selects index 0
        chosen, rewards = algo.step()
        assert chosen.shape == (n_bandits,)
        assert rewards.shape == (n_bandits,)
        assert np.all(chosen == 0)

        # After one step, only arm 0 should have count 1 for all bandits
        counts = algo.arm_pull_counts
        assert np.all(counts[:, 0] == 1)
        assert np.all(counts[:, 1:] == 0)

        log("TestGreedyAlwaysArm0Initially produced data:", LOG_FILE)
        log("chosen preview: " + str(chosen[:10].tolist()), LOG_FILE)
        log("rewards summary: " + summarize_array(rewards), LOG_FILE)
        log("counts row0: " + str(counts[0].tolist()), LOG_FILE)
        log("", LOG_FILE)


class TestEpsFixedEps0BehavesGreedy:
    def run(self):
        log("Running TestEpsFixedEps0BehavesGreedy", LOG_FILE)
        np.random.seed(2)

        n_bandits = 3000
        n_arms = 6

        gang = Gang_of_Bandits(n_bandits=n_bandits, n_arms=n_arms, distribution="bernoulli")
        algo = EpsilonGreedyFixedBulkAlgorithm(gang, epsilon=0.0)

        for _ in range(10):
            chosen, rewards = algo.step()
            assert chosen.shape == (n_bandits,)
            assert rewards.shape == (n_bandits,)
            assert np.all(chosen == 0)

        counts = algo.arm_pull_counts
        assert np.all(counts[:, 0] == 10)
        assert np.all(counts[:, 1:] == 0)

        log("TestEpsFixedEps0BehavesGreedy produced data:", LOG_FILE)
        log("counts row0: " + str(counts[0].tolist()), LOG_FILE)
        log("counts summary: " + summarize_array(counts), LOG_FILE)
        log("", LOG_FILE)


class TestEpsFixedEps1Randomness:
    def run(self):
        log("Running TestEpsFixedEps1Randomness", LOG_FILE)
        np.random.seed(3)

        n_bandits = 5000
        n_arms = 8

        gang = Gang_of_Bandits(n_bandits=n_bandits, n_arms=n_arms, distribution="gaussian")
        algo = EpsilonGreedyFixedBulkAlgorithm(gang, epsilon=1.0)

        chosen, rewards = algo.step()
        assert chosen.shape == (n_bandits,)
        assert rewards.shape == (n_bandits,)

        # With epsilon=1, all bandits explore (random arm). With many bandits,
        # we should see > 1 unique chosen arm with overwhelming probability.
        assert np.unique(chosen).size > 1

        # Counts should sum to 1 per bandit after one step
        assert np.all(algo.arm_pull_counts.sum(axis=1) == 1)

        log("TestEpsFixedEps1Randomness produced data:", LOG_FILE)
        log("unique_arms_count=" + str(int(np.unique(chosen).size)), LOG_FILE)
        log("chosen preview: " + str(chosen[:20].tolist()), LOG_FILE)
        log("rewards summary: " + summarize_array(rewards), LOG_FILE)
        log("", LOG_FILE)


class TestStatisticalSanityBestArmDominates:
    def run(self):
        log("Running TestStatisticalSanityBestArmDominates", LOG_FILE)
        np.random.seed(5)

        # Gang_of_Bandits sorts means descending => true best arm index is 0
        n_bandits = 10000
        n_arms = 10
        steps = 200

        gang = Gang_of_Bandits(n_bandits=n_bandits, n_arms=n_arms, distribution="gaussian")
        algo = EpsilonGreedyFixedBulkAlgorithm(gang, epsilon=0.1)

        for _ in range(steps):
            algo.step()

        # For most bandits, the learned argmax should be arm 0
        learned_best = np.argmax(algo.arm_value_estimates, axis=1)
        hit_rate = float((learned_best == 0).mean())

        # Sanity threshold: should be well above random baseline (0.1)
        # Keep loose to avoid flaky runs.
        assert hit_rate > 0.30

        log("TestStatisticalSanityBestArmDominates produced data:", LOG_FILE)
        log("learned_best summary: " + summarize_array(learned_best), LOG_FILE)
        log(f"hit_rate={hit_rate:.4f} (random baseline={1.0/n_arms:.4f})", LOG_FILE)
        log("", LOG_FILE)


def run_all_tests():
    tests = [
        TestInitShapesGreedy(),
        TestGreedyAlwaysArm0Initially(),
        TestEpsFixedEps0BehavesGreedy(),
        TestEpsFixedEps1Randomness(),
        TestStatisticalSanityBestArmDominates(),
    ]

    for test in tests:
        name = test.__class__.__name__
        try:
            test.run()
            log(f"{name}: PASS", LOG_FILE)
        except Exception as e:
            log(f"{name}: FAIL -> {type(e).__name__}: {e}", LOG_FILE)
            raise

    log("All greedy bulk algorithm tests completed successfully.", LOG_FILE)


if __name__ == "__main__":
    run_all_tests()
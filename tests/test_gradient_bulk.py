# tests/test_gradient_bulk.py

import numpy as np

from src.bandits import Gang_of_Bandits
from src.gradient import (
    PolicyGradientBulkAlgorithm,
    PolicyGradientBaselineBulkAlgorithm,
)
from util.io_helpers import log, out


LOG_FILE = "policy_gradient_bulk_test.log"


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


class TestInitShapesPolicyGradient:
    def run(self):
        log("Running TestInitShapesPolicyGradient", LOG_FILE)
        np.random.seed(0)

        gang = Gang_of_Bandits(n_bandits=1000, n_arms=7, distribution="gaussian")
        algo = PolicyGradientBulkAlgorithm(gang, alpha=0.05)

        assert algo.theta.shape == (1000, 7)
        assert algo.arm_pull_counts.shape == (1000, 7)
        assert algo.arm_reward_sums.shape == (1000, 7)
        assert algo.total_steps == 0

        log("TestInitShapesPolicyGradient produced data:", LOG_FILE)
        log("theta summary: " + summarize_array(algo.theta), LOG_FILE)
        log("counts summary: " + summarize_array(algo.arm_pull_counts), LOG_FILE)
        log("", LOG_FILE)


class TestInitShapesPolicyGradientBaseline:
    def run(self):
        log("Running TestInitShapesPolicyGradientBaseline", LOG_FILE)
        np.random.seed(0)

        gang = Gang_of_Bandits(n_bandits=1000, n_arms=7, distribution="gaussian")
        algo = PolicyGradientBaselineBulkAlgorithm(gang, alpha=0.05)

        assert algo.theta.shape == (1000, 7)
        assert algo.arm_pull_counts.shape == (1000, 7)
        assert algo.arm_reward_sums.shape == (1000, 7)
        assert algo.baseline.shape == (1000,)
        assert algo.total_steps == 0

        log("TestInitShapesPolicyGradientBaseline produced data:", LOG_FILE)
        log("theta summary: " + summarize_array(algo.theta), LOG_FILE)
        log("baseline summary: " + summarize_array(algo.baseline), LOG_FILE)
        log("", LOG_FILE)


class TestStepShapesAndCountsUpdate:
    def run(self):
        log("Running TestStepShapesAndCountsUpdate", LOG_FILE)
        np.random.seed(1)

        n_bandits = 3000
        n_arms = 6
        gang = Gang_of_Bandits(n_bandits=n_bandits, n_arms=n_arms, distribution="bernoulli")

        algo = PolicyGradientBulkAlgorithm(gang, alpha=0.1)
        chosen, rewards = algo.step()

        assert chosen.shape == (n_bandits,)
        assert rewards.shape == (n_bandits,)
        assert np.all((chosen >= 0) & (chosen < n_arms))
        assert algo.total_steps == 1

        # Exactly one pull per bandit after one step
        assert np.all(algo.arm_pull_counts.sum(axis=1) == 1)

        # Theta should have changed from all-zeros for (almost surely) at least some bandits
        assert np.any(algo.theta != 0.0)

        log("TestStepShapesAndCountsUpdate produced data:", LOG_FILE)
        log("chosen preview: " + str(chosen[:20].tolist()), LOG_FILE)
        log("rewards summary: " + summarize_array(rewards), LOG_FILE)
        log("counts row0: " + str(algo.arm_pull_counts[0].tolist()), LOG_FILE)
        log("theta row0 preview: " + str(algo.theta[0][:10].tolist()), LOG_FILE)
        log("", LOG_FILE)


class TestBaselineUpdatesMonotonicallyByMean:
    def run(self):
        log("Running TestBaselineUpdatesMonotonicallyByMean", LOG_FILE)
        np.random.seed(2)

        n_bandits = 2000
        n_arms = 5
        gang = Gang_of_Bandits(n_bandits=n_bandits, n_arms=n_arms, distribution="gaussian")

        algo = PolicyGradientBaselineBulkAlgorithm(gang, alpha=0.05)

        baselines = []
        for _ in range(20):
            _, rewards = algo.step()
            baselines.append(algo.baseline.copy())

        baselines = np.stack(baselines, axis=0)  # (T, B)

        # Baseline should always be finite
        assert np.all(np.isfinite(baselines))

        # Baseline at step t equals running mean of rewards (per bandit).
        # We do a weak check: final baseline close to mean of observed rewards (per bandit),
        # computed from arm_reward_sums (which stores per-arm sums).
        total_reward = algo.arm_reward_sums.sum(axis=1)  # (B,)
        expected = total_reward / float(algo.total_steps)
        err = np.abs(algo.baseline - expected)

        assert float(np.mean(err)) < 1e-9

        log("TestBaselineUpdatesMonotonicallyByMean produced data:", LOG_FILE)
        log("final baseline summary: " + summarize_array(algo.baseline), LOG_FILE)
        log("mean abs error vs expected: " + str(float(np.mean(err))), LOG_FILE)
        log("", LOG_FILE)


class TestStatisticalSanityPolicyGradientFindsBestArm:
    def run(self):
        log("Running TestStatisticalSanityPolicyGradientFindsBestArm", LOG_FILE)
        np.random.seed(3)

        # Gang_of_Bandits sorts means descending => best arm index is 0
        n_bandits = 12000
        n_arms = 10
        steps = 600

        gang = Gang_of_Bandits(n_bandits=n_bandits, n_arms=n_arms, distribution="bernoulli")
        algo = PolicyGradientBulkAlgorithm(gang, alpha=0.05)

        for _ in range(steps):
            algo.step()

        # Evaluate learned preference via argmax(theta)
        learned_best = np.argmax(algo.theta, axis=1)
        hit_rate = float((learned_best == 0).mean())

        # Loose threshold; should beat random (0.1)
        assert hit_rate > 0.20

        log("TestStatisticalSanityPolicyGradientFindsBestArm produced data:", LOG_FILE)
        log("learned_best summary: " + summarize_array(learned_best), LOG_FILE)
        log(f"hit_rate={hit_rate:.4f} (random baseline={1.0/n_arms:.4f})", LOG_FILE)
        log("", LOG_FILE)


class TestStatisticalSanityPolicyGradientBaselineFindsBestArm:
    def run(self):
        log("Running TestStatisticalSanityPolicyGradientBaselineFindsBestArm", LOG_FILE)
        np.random.seed(3)

        n_bandits = 12000
        n_arms = 10
        steps = 600

        gang = Gang_of_Bandits(n_bandits=n_bandits, n_arms=n_arms, distribution="bernoulli")
        algo = PolicyGradientBaselineBulkAlgorithm(gang, alpha=0.05)

        for _ in range(steps):
            algo.step()

        learned_best = np.argmax(algo.theta, axis=1)
        hit_rate = float((learned_best == 0).mean())

        assert hit_rate > 0.20

        log("TestStatisticalSanityPolicyGradientBaselineFindsBestArm produced data:", LOG_FILE)
        log("learned_best summary: " + summarize_array(learned_best), LOG_FILE)
        log(f"hit_rate={hit_rate:.4f} (random baseline={1.0/n_arms:.4f})", LOG_FILE)
        log("", LOG_FILE)


def run_all_tests():
    tests = [
        TestInitShapesPolicyGradient(),
        TestInitShapesPolicyGradientBaseline(),
        TestStepShapesAndCountsUpdate(),
        TestBaselineUpdatesMonotonicallyByMean(),
        TestStatisticalSanityPolicyGradientFindsBestArm(),
        TestStatisticalSanityPolicyGradientBaselineFindsBestArm(),
    ]

    for test in tests:
        name = test.__class__.__name__
        try:
            test.run()
            log(f"{name}: PASS", LOG_FILE)
        except Exception as e:
            log(f"{name}: FAIL -> {type(e).__name__}: {e}", LOG_FILE)
            raise

    log("All policy gradient bulk algorithm tests completed successfully.", LOG_FILE)


if __name__ == "__main__":
    run_all_tests()
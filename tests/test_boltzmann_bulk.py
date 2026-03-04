# tests/test_boltzmann_bulk.py

import numpy as np

from src.bandits import Gang_of_Bandits
from src.boltzmann import (
    BoltzmannExplorationBulkAlgorithm,
    BoltzmannGumbelTrickBulkAlgorithm,
    BoltzmannArbitraryNoiseBulkAlgorithm,
    GumbelScaledBonusBulkAlgorithm,
)
from util.io_helpers import log, out


LOG_FILE = "boltzmann_bulk_test.log"

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


class TestInitShapesBoltzmannExploration:
    def run(self):
        log("Running TestInitShapesBoltzmannExploration", LOG_FILE)
        np.random.seed(0)

        gang = Gang_of_Bandits(n_bandits=1000, n_arms=7, distribution="gaussian")
        algo = BoltzmannExplorationBulkAlgorithm(gang, theta=1.5)

        assert algo.arm_pull_counts.shape == (1000, 7)
        assert algo.arm_value_estimates.shape == (1000, 7)
        assert algo.total_steps == 0

        log("TestInitShapesBoltzmannExploration produced data:", LOG_FILE)
        log("means summary: " + summarize_array(gang.means), LOG_FILE)
        log("counts summary: " + summarize_array(algo.arm_pull_counts), LOG_FILE)
        log("estimates summary: " + summarize_array(algo.arm_value_estimates), LOG_FILE)
        log("", LOG_FILE)


class TestInitShapesBoltzmannGumbelTrick:
    def run(self):
        log("Running TestInitShapesBoltzmannGumbelTrick", LOG_FILE)
        np.random.seed(0)

        gang = Gang_of_Bandits(n_bandits=1000, n_arms=7, distribution="gaussian")
        algo = BoltzmannGumbelTrickBulkAlgorithm(gang, theta=1.5)

        assert algo.arm_pull_counts.shape == (1000, 7)
        assert algo.arm_value_estimates.shape == (1000, 7)
        assert algo.total_steps == 0

        log("TestInitShapesBoltzmannGumbelTrick produced data:", LOG_FILE)
        log("means summary: " + summarize_array(gang.means), LOG_FILE)
        log("counts summary: " + summarize_array(algo.arm_pull_counts), LOG_FILE)
        log("estimates summary: " + summarize_array(algo.arm_value_estimates), LOG_FILE)
        log("", LOG_FILE)


class TestInitShapesBoltzmannArbitraryNoise:
    def run(self):
        log("Running TestInitShapesBoltzmannArbitraryNoise", LOG_FILE)
        np.random.seed(0)

        gang = Gang_of_Bandits(n_bandits=1000, n_arms=7, distribution="gaussian")
        algo = BoltzmannArbitraryNoiseBulkAlgorithm(gang, theta=1.5, noise="gumbel")

        assert algo.arm_pull_counts.shape == (1000, 7)
        assert algo.arm_value_estimates.shape == (1000, 7)
        assert algo.total_steps == 0

        log("TestInitShapesBoltzmannArbitraryNoise produced data:", LOG_FILE)
        log("means summary: " + summarize_array(gang.means), LOG_FILE)
        log("counts summary: " + summarize_array(algo.arm_pull_counts), LOG_FILE)
        log("estimates summary: " + summarize_array(algo.arm_value_estimates), LOG_FILE)
        log("", LOG_FILE)


class TestInitShapesGumbelScaledBonus:
    def run(self):
        log("Running TestInitShapesGumbelScaledBonus", LOG_FILE)
        np.random.seed(0)

        gang = Gang_of_Bandits(n_bandits=1000, n_arms=7, distribution="gaussian")
        algo = GumbelScaledBonusBulkAlgorithm(gang, C=1.0)

        assert algo.arm_pull_counts.shape == (1000, 7)
        assert algo.arm_value_estimates.shape == (1000, 7)
        assert algo.total_steps == 0

        log("TestInitShapesGumbelScaledBonus produced data:", LOG_FILE)
        log("means summary: " + summarize_array(gang.means), LOG_FILE)
        log("counts summary: " + summarize_array(algo.arm_pull_counts), LOG_FILE)
        log("estimates summary: " + summarize_array(algo.arm_value_estimates), LOG_FILE)
        log("", LOG_FILE)


class TestBoltzmannStartsUniformish:
    def run(self):
        log("Running TestBoltzmannStartsUniformish", LOG_FILE)
        np.random.seed(1)

        n_bandits = 8000
        n_arms = 10
        gang = Gang_of_Bandits(n_bandits=n_bandits, n_arms=n_arms, distribution="gaussian")

        # With all Q_hat equal (0), selection should be effectively uniform.
        algo1 = BoltzmannExplorationBulkAlgorithm(gang, theta=1.0)
        chosen1, rewards1 = algo1.step()

        algo2 = BoltzmannGumbelTrickBulkAlgorithm(gang, theta=1.0)
        chosen2, rewards2 = algo2.step()

        assert chosen1.shape == (n_bandits,)
        assert chosen2.shape == (n_bandits,)
        assert rewards1.shape == (n_bandits,)
        assert rewards2.shape == (n_bandits,)

        # With many bandits, should see multiple distinct arms chosen.
        assert np.unique(chosen1).size > 1
        assert np.unique(chosen2).size > 1

        log("TestBoltzmannStartsUniformish produced data:", LOG_FILE)
        log("unique_arms_softmax=" + str(int(np.unique(chosen1).size)), LOG_FILE)
        log("unique_arms_gumbel=" + str(int(np.unique(chosen2).size)), LOG_FILE)
        log("chosen1 preview: " + str(chosen1[:20].tolist()), LOG_FILE)
        log("chosen2 preview: " + str(chosen2[:20].tolist()), LOG_FILE)
        log("", LOG_FILE)


class TestArbitraryNoiseBetaWorksAndUnknownRaises:
    def run(self):
        log("Running TestArbitraryNoiseBetaWorksAndUnknownRaises", LOG_FILE)
        np.random.seed(2)

        n_bandits = 5000
        n_arms = 8
        gang = Gang_of_Bandits(n_bandits=n_bandits, n_arms=n_arms, distribution="gaussian")

        algo = BoltzmannArbitraryNoiseBulkAlgorithm(
            gang,
            theta=1.0,
            noise="beta",
            noise_params={"a": 2.0, "b": 5.0},
        )

        chosen, rewards = algo.step()
        assert chosen.shape == (n_bandits,)
        assert rewards.shape == (n_bandits,)
        assert np.all((chosen >= 0) & (chosen < n_arms))

        # Unknown noise should raise ValueError
        try:
            bad = BoltzmannArbitraryNoiseBulkAlgorithm(gang, theta=1.0, noise="not_a_real_noise")
            bad.step()
            raise AssertionError("Expected ValueError for unknown noise, but no error was raised.")
        except ValueError:
            pass

        log("TestArbitraryNoiseBetaWorksAndUnknownRaises produced data:", LOG_FILE)
        log("unique_arms_beta=" + str(int(np.unique(chosen).size)), LOG_FILE)
        log("chosen preview: " + str(chosen[:20].tolist()), LOG_FILE)
        log("", LOG_FILE)


class TestGumbelScaledBonusTriesAllArmsFirstKSteps:
    def run(self):
        log("Running TestGumbelScaledBonusTriesAllArmsFirstKSteps", LOG_FILE)
        np.random.seed(3)

        n_bandits = 2000
        n_arms = 6
        gang = Gang_of_Bandits(n_bandits=n_bandits, n_arms=n_arms, distribution="gaussian")
        algo = GumbelScaledBonusBulkAlgorithm(gang, C=1.0)

        chosen_history = []
        for _ in range(n_arms):
            chosen, rewards = algo.step()
            assert chosen.shape == (n_bandits,)
            assert rewards.shape == (n_bandits,)
            chosen_history.append(chosen)

        # With +inf for unpulled arms and argmax tie-break, should pick 0,1,2,...,K-1
        for t, chosen in enumerate(chosen_history):
            assert np.all(chosen == t), f"At t={t}, expected all chosen=={t}"

        counts = algo.arm_pull_counts
        assert np.all(counts.sum(axis=1) == n_arms)
        assert np.all(counts == 1)

        log("TestGumbelScaledBonusTriesAllArmsFirstKSteps produced data:", LOG_FILE)
        log("counts row0: " + str(counts[0].tolist()), LOG_FILE)
        log("", LOG_FILE)


class TestStatisticalSanityGumbelScaledBonusFindsBestArm:
    def run(self):
        log("Running TestStatisticalSanityGumbelScaledBonusFindsBestArm", LOG_FILE)
        np.random.seed(4)

        # Gang_of_Bandits sorts means descending => true best arm index is 0
        n_bandits = 10000
        n_arms = 10
        steps = 300

        gang = Gang_of_Bandits(n_bandits=n_bandits, n_arms=n_arms, distribution="gaussian")
        algo = GumbelScaledBonusBulkAlgorithm(gang, C=1.0)

        for _ in range(steps):
            algo.step()

        learned_best = np.argmax(algo.arm_value_estimates, axis=1)
        hit_rate = float((learned_best == 0).mean())

        # Loose threshold: should beat random baseline
        assert hit_rate > 0.25

        log("TestStatisticalSanityGumbelScaledBonusFindsBestArm produced data:", LOG_FILE)
        log("learned_best summary: " + summarize_array(learned_best), LOG_FILE)
        log(f"hit_rate={hit_rate:.4f} (random baseline={1.0/n_arms:.4f})", LOG_FILE)
        log("", LOG_FILE)


def run_all_tests():
    tests = [
        TestInitShapesBoltzmannExploration(),
        TestInitShapesBoltzmannGumbelTrick(),
        TestInitShapesBoltzmannArbitraryNoise(),
        TestInitShapesGumbelScaledBonus(),
        TestBoltzmannStartsUniformish(),
        TestArbitraryNoiseBetaWorksAndUnknownRaises(),
        TestGumbelScaledBonusTriesAllArmsFirstKSteps(),
        TestStatisticalSanityGumbelScaledBonusFindsBestArm(),
    ]

    for test in tests:
        name = test.__class__.__name__
        try:
            test.run()
            log(f"{name}: PASS", LOG_FILE)
        except Exception as e:
            log(f"{name}: FAIL -> {type(e).__name__}: {e}", LOG_FILE)
            raise

    log("All boltzmann bulk algorithm tests completed successfully.", LOG_FILE)


if __name__ == "__main__":
    run_all_tests()
import numpy as np

from src.bandits import Gang_of_Bandits
from src.etc import ETCBulkAlgorithm
from util.io_helpers import log, out


LOG_FILE = "etc_bulk_test.log"


def summarize_array(arr, n_preview=8):
    arr = np.asarray(arr)
    if arr.size == 0:
        return f"shape={arr.shape}, empty"
    preview = arr.ravel()[:n_preview]
    # Safe numeric summary
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


class TestInitShapes:
    def run(self):
        log("Running TestInitShapes", LOG_FILE)
        np.random.seed(0)

        gang = Gang_of_Bandits(n_bandits=1000, n_arms=7, distribution="gaussian")
        algo = ETCBulkAlgorithm(gang, exploration_rounds=5)

        assert algo.arm_pull_counts.shape == (1000, 7)
        assert algo.arm_reward_sums.shape == (1000, 7)
        assert algo.committed_arm.shape == (1000,)
        assert algo.total_steps == 0
        assert algo.committed is False

        log("TestInitShapes produced data:", LOG_FILE)
        log(f"exploration_steps_total={algo.exploration_steps_total}", LOG_FILE)
        log("means summary: " + summarize_array(gang.means), LOG_FILE)
        log("arm_pull_counts summary: " + summarize_array(algo.arm_pull_counts), LOG_FILE)
        log("arm_reward_sums summary: " + summarize_array(algo.arm_reward_sums), LOG_FILE)
        log("committed_arm summary: " + summarize_array(algo.committed_arm), LOG_FILE)
        log("", LOG_FILE)


class TestExplorationRoundRobin:
    def run(self):
        log("Running TestExplorationRoundRobin", LOG_FILE)
        np.random.seed(1)

        n_bandits = 2000
        n_arms = 5
        exploration_rounds = 3

        gang = Gang_of_Bandits(n_bandits=n_bandits, n_arms=n_arms, distribution="gaussian")
        algo = ETCBulkAlgorithm(gang, exploration_rounds=exploration_rounds)

        # During exploration, chosen arm should be (t mod K) for ALL bandits
        chosen_history = []
        for t in range(algo.exploration_steps_total):
            chosen, rewards = algo.step()
            expected_arm = t % n_arms

            assert chosen.shape == (n_bandits,)
            assert rewards.shape == (n_bandits,)
            assert np.all(chosen == expected_arm)

            # Store a tiny sample for output
            chosen_history.append(int(chosen[0]))

        log("TestExplorationRoundRobin produced data:", LOG_FILE)
        log(f"chosen_history_first_bandit={chosen_history}", LOG_FILE)
        log("last_rewards_summary: " + summarize_array(rewards), LOG_FILE)
        log("", LOG_FILE)


class TestExactExplorationCounts:
    def run(self):
        log("Running TestExactExplorationCounts", LOG_FILE)
        np.random.seed(2)

        n_bandits = 3000
        n_arms = 6
        exploration_rounds = 4

        gang = Gang_of_Bandits(n_bandits=n_bandits, n_arms=n_arms, distribution="bernoulli")
        algo = ETCBulkAlgorithm(gang, exploration_rounds=exploration_rounds)

        # Run full exploration
        for _ in range(algo.exploration_steps_total):
            algo.step()

        # Now each arm must have been pulled exactly exploration_rounds times per bandit
        counts = algo.arm_pull_counts
        assert counts.shape == (n_bandits, n_arms)
        assert np.all(counts == exploration_rounds)

        log("TestExactExplorationCounts produced data:", LOG_FILE)
        log("counts row0: " + str(counts[0].tolist()), LOG_FILE)
        log("counts summary: " + summarize_array(counts), LOG_FILE)
        log("", LOG_FILE)


class TestCommitmentAndStickiness:
    def run(self):
        log("Running TestCommitmentAndStickiness", LOG_FILE)
        np.random.seed(3)

        n_bandits = 5000
        n_arms = 8
        exploration_rounds = 3

        gang = Gang_of_Bandits(n_bandits=n_bandits, n_arms=n_arms, distribution="gaussian")
        algo = ETCBulkAlgorithm(gang, exploration_rounds=exploration_rounds)

        # Run exploration fully
        for _ in range(algo.exploration_steps_total):
            algo.step()

        assert algo.committed is False  # commitment occurs on first post-exploration step

        # First step after exploration triggers commitment + uses committed arms
        chosen1, rewards1 = algo.step()
        assert algo.committed is True
        assert algo.committed_arm.shape == (n_bandits,)
        assert np.all((algo.committed_arm >= 0) & (algo.committed_arm < n_arms))
        assert np.array_equal(chosen1, algo.committed_arm)

        # Next few steps must keep choosing the same committed arm per bandit
        for _ in range(5):
            chosen_t, rewards_t = algo.step()
            assert np.array_equal(chosen_t, algo.committed_arm)

        log("TestCommitmentAndStickiness produced data:", LOG_FILE)
        log("committed_arm summary: " + summarize_array(algo.committed_arm), LOG_FILE)
        log("chosen1 preview: " + str(chosen1[:10].tolist()), LOG_FILE)
        log("rewards1 summary: " + summarize_array(rewards1), LOG_FILE)
        log("last_rewards summary: " + summarize_array(rewards_t), LOG_FILE)
        log("", LOG_FILE)


class TestStatisticalSanityBestArmHitRateGaussian:
    def run(self):
        log("Running TestStatisticalSanityBestArmHitRateGaussian", LOG_FILE)
        np.random.seed(4)

        # Keep this reasonable runtime but meaningful signal
        n_bandits = 10000
        n_arms = 10
        exploration_rounds = 20  # more exploration => better chance to identify best arm

        gang = Gang_of_Bandits(n_bandits=n_bandits, n_arms=n_arms, distribution="gaussian")
        algo = ETCBulkAlgorithm(gang, exploration_rounds=exploration_rounds)

        # True best arms by mean
        true_best = np.argmax(gang.means, axis=1)

        # Run exploration and commit
        for _ in range(algo.exploration_steps_total):
            algo.step()
        algo.step()  # triggers commit

        committed = algo.committed_arm
        hit_rate = float((committed == true_best).mean())

        # Sanity threshold: should beat random guessing (=1/n_arms=0.1) by a decent margin.
        # This is intentionally not too strict to avoid flaky tests across RNG/platforms.
        assert hit_rate > 0.20

        log("TestStatisticalSanityBestArmHitRateGaussian produced data:", LOG_FILE)
        log("true_best summary: " + summarize_array(true_best), LOG_FILE)
        log("committed summary: " + summarize_array(committed), LOG_FILE)
        log(f"hit_rate={hit_rate:.4f} (random baseline={1.0/n_arms:.4f})", LOG_FILE)
        log("", LOG_FILE)


def run_all_tests():
    tests = [
        TestInitShapes(),
        TestExplorationRoundRobin(),
        TestExactExplorationCounts(),
        TestCommitmentAndStickiness(),
        TestStatisticalSanityBestArmHitRateGaussian(),
    ]

    for test in tests:
        name = test.__class__.__name__
        try:
            test.run()
            log(f"{name}: PASS", LOG_FILE)
        except Exception as e:
            log(f"{name}: FAIL -> {type(e).__name__}: {e}", LOG_FILE)
            raise

    log("All ETCBulkAlgorithm tests completed successfully.", LOG_FILE)


if __name__ == "__main__":
    run_all_tests()
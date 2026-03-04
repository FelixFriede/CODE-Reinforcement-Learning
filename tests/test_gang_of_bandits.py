import numpy as np

from src.bandits import Gang_of_Bandits
from util.io_helpers import log, out


LOG_FILE = "gang_bandits_test.log"
OUT_FILE = "gang_bandits_test.txt"


def summarize_array(arr, n_preview=5):
    """Return short textual summary of numpy array."""
    arr = np.asarray(arr)

    preview = arr[:n_preview]
    return (
        f"shape={arr.shape}, "
        f"mean={arr.mean():.4f}, std={arr.std():.4f}, "
        f"min={arr.min():.4f}, max={arr.max():.4f}, "
        f"preview={preview.tolist()}"
    )


class TestGangBanditsInitialization:

    def run(self):
        log("Running initialization test", LOG_FILE)

        n_bandits = 1000
        n_arms = 10

        gang = Gang_of_Bandits(n_bandits, n_arms, "gaussian")

        assert gang.means.shape == (n_bandits, n_arms)

        out("Initialization Test", OUT_FILE)
        out(summarize_array(gang.means.flatten()), OUT_FILE)
        out("", OUT_FILE)


class TestGaussianBulkPull:

    def run(self):
        log("Running gaussian bulk pull test", LOG_FILE)

        n_bandits = 1000
        n_arms = 5

        gang = Gang_of_Bandits(n_bandits, n_arms, "gaussian")

        rewards = gang.bulk_pull(2)

        assert rewards.shape == (n_bandits,)

        out("Gaussian Bulk Pull Test", OUT_FILE)
        out(summarize_array(rewards), OUT_FILE)
        out("", OUT_FILE)


class TestBernoulliBulkPull:

    def run(self):
        log("Running bernoulli bulk pull test", LOG_FILE)

        n_bandits = 2000
        n_arms = 5

        gang = Gang_of_Bandits(n_bandits, n_arms, "bernoulli")

        rewards = gang.bulk_pull(1)

        assert rewards.shape == (n_bandits,)
        assert np.all((rewards == 0) | (rewards == 1))

        ones_ratio = rewards.mean()

        out("Bernoulli Bulk Pull Test", OUT_FILE)
        out(summarize_array(rewards), OUT_FILE)
        out(f"fraction_of_ones={ones_ratio:.4f}", OUT_FILE)
        out("", OUT_FILE)


class TestPerBanditArmSelection:

    def run(self):
        log("Running per-bandit arm selection test", LOG_FILE)

        n_bandits = 1000
        n_arms = 4

        gang = Gang_of_Bandits(n_bandits, n_arms, "gaussian")

        arms = np.random.randint(0, n_arms, size=n_bandits)

        rewards = gang.bulk_pull(arms)

        assert rewards.shape == (n_bandits,)

        out("Per-Bandit Arm Selection Test", OUT_FILE)
        out("Arms chosen summary:", OUT_FILE)
        out(summarize_array(arms), OUT_FILE)
        out("Rewards summary:", OUT_FILE)
        out(summarize_array(rewards), OUT_FILE)
        out("", OUT_FILE)


class TestStatisticalSanity:

    def run(self):
        log("Running statistical sanity test", LOG_FILE)

        n_bandits = 1000
        n_arms = 3
        pulls = 5000

        gang = Gang_of_Bandits(n_bandits, n_arms, "gaussian")

        arm = 1
        mu = gang.means[:, arm]

        rewards = np.zeros((pulls, n_bandits))

        for i in range(pulls):
            rewards[i] = gang.bulk_pull(arm)

        empirical = rewards.mean(axis=0)

        error = np.mean(np.abs(empirical - mu))

        out("Statistical Sanity Test", OUT_FILE)
        out("True means summary:", OUT_FILE)
        out(summarize_array(mu), OUT_FILE)
        out("Empirical means summary:", OUT_FILE)
        out(summarize_array(empirical), OUT_FILE)
        out(f"mean_absolute_error={error:.6f}", OUT_FILE)
        out("", OUT_FILE)


class TestLargeBatchPerformance:

    def run(self):
        log("Running large batch performance test", LOG_FILE)

        n_bandits = 100000
        n_arms = 10

        gang = Gang_of_Bandits(n_bandits, n_arms, "gaussian")

        rewards = gang.bulk_pull(3)

        assert rewards.shape == (n_bandits,)

        out("Large Batch Performance Test", OUT_FILE)
        out(summarize_array(rewards), OUT_FILE)
        out("", OUT_FILE)


def run_all_tests():

    tests = [
        TestGangBanditsInitialization(),
        TestGaussianBulkPull(),
        TestBernoulliBulkPull(),
        TestPerBanditArmSelection(),
        TestStatisticalSanity(),
        TestLargeBatchPerformance(),
    ]

    for test in tests:
        name = test.__class__.__name__

        try:
            test.run()
            out(f"{name}: PASS", OUT_FILE)
        except Exception as e:
            out(f"{name}: FAIL -> {e}", OUT_FILE)
            raise

    out("All Gang_of_Bandits tests completed successfully.", OUT_FILE)


if __name__ == "__main__":
    run_all_tests()
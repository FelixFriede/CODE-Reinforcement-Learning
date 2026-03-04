# script/test_etc.py

from src.bandits import Bandit
from src.etc import ETCAlgorithm
from util.io_helpers import log, out


def test_etc_algorithm():
# Simple functional test for the ETC algorithm.

    # Create a bandit with known means
    bandit = Bandit(
        n_arms=3,
        distribution="bernoulli",
        means=[0.2, 0.5, 0.8]
    )

    # Each arm explored 5 times
    etc = ETCAlgorithm(bandit=bandit, exploration_rounds=5)
    results = []

    # Run for a fixed number of steps
    for _ in range(25):
        arm, reward = etc.step()
        results.append((arm, reward))

    out(f"ETC results: {results}", "etc_output.txt")


if __name__ == "__main__":
    test_etc_algorithm()
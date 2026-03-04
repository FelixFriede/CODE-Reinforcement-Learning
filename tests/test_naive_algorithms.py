# script/test_naive_algorithms.py

from src.bandits import Bandit

from src.etc import ETCAlgorithm

from src.greedy import (
    GreedyAlgorithm,
    EpsilonGreedyFixedAlgorithm,
    EpsilonGreedyDecreasingAlgorithm,
)

from src.ucb import (
    UCBAlgorithm,
    UCBSubGaussianAlgorithm,
)

from src.boltzmann import (
    BoltzmannExplorationAlgorithm,
    BoltzmannGumbelTrickAlgorithm,
    BoltzmannArbitraryNoiseAlgorithm,
    GumbelScaledBonusAlgorithm,
)

from src.gradient import (
    PolicyGradientAlgorithm,
    PolicyGradientBaselineAlgorithm,
)

from util.io_helpers import log


def _make_bandit(distribution: str):
    means = [0.2, 0.5, 0.8]

    if distribution == "bernoulli":
        return Bandit(
            n_arms=3,
            distribution="bernoulli",
            means=means
        )

    try:
        return Bandit(
            n_arms=3,
            distribution="gaussian",
            means=means,
            sigma=1.0
        )
    except TypeError:
        pass

    try:
        return Bandit(
            n_arms=3,
            distribution="gaussian",
            means=means,
            std=1.0
        )
    except TypeError:
        pass

    return Bandit(
        n_arms=3,
        distribution="gaussian",
        means=means
    )


def _run_steps(algo, n_steps: int):
    results = []
    for _ in range(n_steps):
        arm, reward = algo.step()
        results.append((arm, reward))
    return results


def test_all_algorithms_for_distribution(distribution: str):

    log(f"=== Testing algorithms with {distribution} bandit ===", "all_algorithms.log")

    # ETC
    bandit = _make_bandit(distribution)
    etc = ETCAlgorithm(bandit=bandit, exploration_rounds=5)
    results = _run_steps(etc, 25)
    log(f"{distribution} ETC results: {results}", "all_algorithms.log")

    # Greedy
    bandit = _make_bandit(distribution)
    greedy = GreedyAlgorithm(bandit=bandit)
    results = _run_steps(greedy, 25)
    log(f"{distribution} Greedy results: {results}", "all_algorithms.log")

    # epsilon-greedy fixed
    bandit = _make_bandit(distribution)
    eg_fixed = EpsilonGreedyFixedAlgorithm(bandit=bandit, epsilon=0.1)
    results = _run_steps(eg_fixed, 25)
    log(f"{distribution} EpsilonGreedyFixed results: {results}", "all_algorithms.log")

    # epsilon-greedy decreasing
    bandit = _make_bandit(distribution)
    eg_dec = EpsilonGreedyDecreasingAlgorithm(bandit=bandit, epsilon0=1.0)
    results = _run_steps(eg_dec, 25)
    log(f"{distribution} EpsilonGreedyDecreasing results: {results}", "all_algorithms.log")

    # UCB
    bandit = _make_bandit(distribution)
    ucb = UCBAlgorithm(bandit=bandit, delta=0.1)
    results = _run_steps(ucb, 25)
    log(f"{distribution} UCB results: {results}", "all_algorithms.log")

    # UCB subgaussian
    bandit = _make_bandit(distribution)
    ucb_sg = UCBSubGaussianAlgorithm(bandit=bandit, delta=0.1, sigma=1.0)
    results = _run_steps(ucb_sg, 25)
    log(f"{distribution} UCBSubGaussian results: {results}", "all_algorithms.log")

    # Boltzmann
    bandit = _make_bandit(distribution)
    boltz = BoltzmannExplorationAlgorithm(bandit=bandit, theta=2.0)
    results = _run_steps(boltz, 25)
    log(f"{distribution} Boltzmann results: {results}", "all_algorithms.log")

    # Boltzmann Gumbel trick
    bandit = _make_bandit(distribution)
    boltz_g = BoltzmannGumbelTrickAlgorithm(bandit=bandit, theta=2.0)
    results = _run_steps(boltz_g, 25)
    log(f"{distribution} BoltzmannGumbel results: {results}", "all_algorithms.log")

    # Boltzmann arbitrary noise (Cauchy)
    bandit = _make_bandit(distribution)
    boltz_cauchy = BoltzmannArbitraryNoiseAlgorithm(
        bandit=bandit,
        theta=2.0,
        noise="cauchy",
        noise_params={"loc": 0.0, "scale": 1.0}
    )
    results = _run_steps(boltz_cauchy, 25)
    log(f"{distribution} BoltzmannArbitraryNoise (Cauchy) results: {results}", "all_algorithms.log")

    # Boltzmann arbitrary noise (Beta)
    bandit = _make_bandit(distribution)
    boltz_beta = BoltzmannArbitraryNoiseAlgorithm(
        bandit=bandit,
        theta=2.0,
        noise="beta",
        noise_params={"a": 2.0, "b": 5.0}
    )
    results = _run_steps(boltz_beta, 25)
    log(f"{distribution} BoltzmannArbitraryNoise (Beta) results: {results}", "all_algorithms.log")

    # Scaled Gumbel bonus
    bandit = _make_bandit(distribution)
    g_scaled = GumbelScaledBonusAlgorithm(bandit=bandit, C=1.0)
    results = _run_steps(g_scaled, 25)
    log(f"{distribution} GumbelScaledBonus results: {results}", "all_algorithms.log")

    # Policy gradient
    bandit = _make_bandit(distribution)
    pg = PolicyGradientAlgorithm(bandit=bandit, alpha=0.1)
    results = _run_steps(pg, 50)
    log(f"{distribution} PolicyGradient results: {results}", "all_algorithms.log")

    # Policy gradient with baseline
    bandit = _make_bandit(distribution)
    pg_b = PolicyGradientBaselineAlgorithm(bandit=bandit, alpha=0.1)
    results = _run_steps(pg_b, 50)
    log(f"{distribution} PolicyGradientBaseline results: {results}", "all_algorithms.log")


def test_all_algorithms():
    test_all_algorithms_for_distribution("bernoulli")
    test_all_algorithms_for_distribution("gaussian")


if __name__ == "__main__":
    test_all_algorithms()
from src.bandits import Bandit

# ------------------------------------------------------------
# Example 1: Gaussian bandit with random means
# ------------------------------------------------------------
bandit1 = Bandit(
    n_arms=5,
    distribution="gaussian"
)

for _ in range(3):
    bandit1.pull(0)


# ------------------------------------------------------------
# Example 2: Bernoulli bandit with manually specified means
# ------------------------------------------------------------
bandit2 = Bandit(
    n_arms=3,
    distribution="bernoulli",
    means=[0.2, 0.5, 0.8]
)

for arm in range(3):
    bandit2.pull(arm)


# ------------------------------------------------------------
# Example 3: Gaussian bandit with gap enforcement
# ------------------------------------------------------------
bandit3 = Bandit(
    n_arms=4,
    distribution="gaussian",
    gap=0.25
)

for _ in range(5):
    bandit3.pull(2)


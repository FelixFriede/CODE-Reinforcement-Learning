# scr/bandits.py

# from util.io_helpers import log, out
# do not let classes log themselves.

import numpy as np


class Bandit:
    """
    Stochastic multi-armed bandit supporting Gaussian and Bernoulli arms.

    Initialization logic (exactly one must apply):
    - If means are provided -> use them
    - Else if gap is provided -> random means + gap enforcement
    - Else -> random means
    """

    # LOG_FILE = "bandit.log"
    # OUT_FILE = "bandit.txt"

    def __init__(self, n_arms, distribution, means=None, gap=None):
        self.n_arms = n_arms
        self.distribution = distribution

        # ---- validate distribution ----
        if distribution not in ("gaussian", "bernoulli"):
            raise ValueError("distribution must be 'gaussian' or 'bernoulli'")

        # ---- validate mean / gap logic ----
        if means is not None and gap is not None:
            raise ValueError("Cannot specify both means and gap")

        if means is not None:
            method = "manual_means"
            self.means = self._init_from_means(means)

        elif gap is not None:
            method = "gap_method"
            self.means = self._init_from_gap(gap)

        else:
            method = "random_means"
            self.means = self._init_random_means()

        # ---- log initialization ----
        # log(f"Initialized Bandit | arms={n_arms}, "f"distribution={distribution}, method={method}, "f"means={self.means.tolist()}",self.LOG_FILE)

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------

    def _init_from_means(self, means):
        means = np.asarray(means, dtype=float)

        if len(means) != self.n_arms:
            raise ValueError("Length of means must equal number of arms")

        if self.distribution == "bernoulli":
            means = np.clip(means, 0.0, 1.0)

        return means

    def _init_random_means(self):
        if self.distribution == "gaussian":
            return np.random.randn(self.n_arms)

        if self.distribution == "bernoulli":
            return np.random.uniform(0.0, 1.0, size=self.n_arms)

    def _init_from_gap(self, gap):
        # Step 1: draw random means
        means = self._init_random_means()

        # Step 2: identify best arm
        sorted_means = np.sort(means)[::-1]
        mu_star = sorted_means[0]

        # Step 3: enforce gap structure
        new_means = np.array(
            [mu_star - k * gap for k in range(self.n_arms)]
        )

        if self.distribution == "bernoulli":
            new_means = np.clip(new_means, 0.0, 1.0)

        return new_means

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def pull(self, arm_index):
        if arm_index < 0 or arm_index >= self.n_arms:
            raise IndexError("Invalid arm index")

        mu = self.means[arm_index]

        if self.distribution == "gaussian":
            reward = np.random.normal(mu, 1.0)

        else:  # bernoulli
            reward = np.random.binomial(1, mu)

        # log(f"Pull | arm={arm_index}, mean={mu}, reward={reward}",self.LOG_FILE)

        return reward
    

import numpy as np


class Gang_of_Bandits:
    """
    Bulk bandit simulator: n_bandits independent bandits, each with n_arms arms.
    Means are stored as a matrix of shape (n_bandits, n_arms).

    bulk_pull supports:
      - arm_index: int -> all bandits pull the same arm
      - arm_index: array-like shape (n_bandits,) -> each bandit pulls its own arm
    """

    def __init__(self, n_bandits, n_arms, distribution):
        self.n_bandits = int(n_bandits)
        self.n_arms = int(n_arms)
        self.distribution = distribution

        if self.n_bandits <= 0 or self.n_arms <= 0:
            raise ValueError("n_bandits and n_arms must be positive integers")

        # ---- validate distribution ----
        if distribution not in ("gaussian", "bernoulli"):
            raise ValueError("distribution must be 'gaussian' or 'bernoulli'")

        self.means = self._init_random_means()  # (n_bandits, n_arms)

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------

    def _init_random_means(self):
        """
        Initialize per-bandit means and sort arms *within each bandit*.

        Sorting is descending (best arm at index 0), which makes arm indices
        comparable across runs for plot (c): arm 0 ~ best, arm 1 ~ 2nd best, etc.
        """
        if self.distribution == "gaussian":
            means = np.random.randn(self.n_bandits, self.n_arms)
        else:  # bernoulli
            means = np.random.uniform(0.0, 1.0, size=(self.n_bandits, self.n_arms))

        # Sort arms within each bandit by mean, best -> worst
        means.sort(axis=1)
        means = means[:, ::-1]

        return means

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def bulk_pull(self, arm_index):
        """
        Return rewards for one pull across all bandits.

        Parameters
        ----------
        arm_index : int or array-like
            - int: all bandits pull that arm
            - array-like of shape (n_bandits,): each bandit pulls its own arm

        Returns
        -------
        rewards : np.ndarray of shape (n_bandits,)
        """

        # Fast path: scalar arm index for all bandits
        if np.isscalar(arm_index):
            a = int(arm_index)
            if a < 0 or a >= self.n_arms:
                raise IndexError("Invalid arm index")

            mu = self.means[:, a]  # shape (n_bandits,)

            if self.distribution == "gaussian":
                # Vectorized: one normal draw per bandit
                return np.random.normal(loc=mu, scale=1.0, size=self.n_bandits)

            # bernoulli: vectorized with per-bandit probabilities
            return (np.random.random(self.n_bandits) < mu).astype(np.int8)


        # General path: one arm per bandit
        arms = np.asarray(arm_index)
        if arms.shape != (self.n_bandits,):
            raise ValueError(f"arm_index must be an int or shape ({self.n_bandits},)")

        if (arms < 0).any() or (arms >= self.n_arms).any():
            raise IndexError("Invalid arm index in arm_index array")

        rows = np.arange(self.n_bandits)
        mu = self.means[rows, arms]  # gather per-bandit chosen-arm mean; shape (n_bandits,)

        if self.distribution == "gaussian":
            return np.random.normal(loc=mu, scale=1.0, size=self.n_bandits)

        return (np.random.random(self.n_bandits) < mu).astype(np.int8)
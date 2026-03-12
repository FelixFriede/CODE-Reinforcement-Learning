# src/boltzmann.py

from __future__ import annotations
from src.bandits import Bandit, Gang_of_Bandits
import numpy as np

try:
    # Optional (used for BetaPrime / Chi if available)
    from scipy import stats as _scipy_stats  # type: ignore
except Exception:
    _scipy_stats = None


# from util.io_helpers import log, out
# do not let classes log themselves.



class BoltzmannExplorationBulkAlgorithm:
    """
    Bulk Boltzmann (softmax) exploration:
        P(A=a) ∝ exp(theta * Q_hat[a])
    Runs one synchronized step over many bandits at once.
    """

    def __init__(self, bandit: Gang_of_Bandits, theta: float):
        self.bandit = bandit
        self.theta = float(theta)

        self.n_bandits = bandit.n_bandits
        self.n_arms = bandit.n_arms

        self.total_steps = 0

        self.arm_pull_counts = np.zeros((self.n_bandits, self.n_arms), dtype=np.int32)
        self.arm_value_estimates = np.zeros((self.n_bandits, self.n_arms), dtype=np.float64)  # Q_hat

    def _sample_softmax_arms(self) -> np.ndarray:
        """
        Sample one arm per bandit using a numerically-stable softmax.
        This uses the Gumbel-max trick to avoid building probabilities explicitly:
            argmax_a (theta*Q_hat[a] + G_a) ~ softmax(theta*Q_hat)
        """
        logits = self.arm_value_estimates * self.theta  # (B, K)
        g = np.random.gumbel(loc=0.0, scale=1.0, size=(self.n_bandits, self.n_arms))
        return np.argmax(logits + g, axis=1).astype(np.int32)

    def _update_estimates(self, chosen_arms: np.ndarray, rewards: np.ndarray) -> None:
        rows = np.arange(self.n_bandits)
        self.arm_pull_counts[rows, chosen_arms] += 1
        pulls = self.arm_pull_counts[rows, chosen_arms].astype(np.float64)

        q_old = self.arm_value_estimates[rows, chosen_arms]
        self.arm_value_estimates[rows, chosen_arms] = q_old + (rewards - q_old) / pulls

    def step(self):
        chosen_arms = self._sample_softmax_arms()
        rewards = self.bandit.bulk_pull(chosen_arms)

        self._update_estimates(chosen_arms, rewards)
        self.total_steps += 1

        return chosen_arms, rewards


# [LEGACY]
class BoltzmannGumbelTrickBulkAlgorithm:
    """
    Bulk Boltzmann exploration via the Gumbel trick:
        A = argmax_a { theta * Q_hat[a] + G_a },  G_a i.i.d. standard Gumbel
    """

    def __init__(self, bandit: Gang_of_Bandits, theta: float):
        self.bandit = bandit
        self.theta = float(theta)

        self.n_bandits = bandit.n_bandits
        self.n_arms = bandit.n_arms

        self.total_steps = 0

        self.arm_pull_counts = np.zeros((self.n_bandits, self.n_arms), dtype=np.int32)
        self.arm_value_estimates = np.zeros((self.n_bandits, self.n_arms), dtype=np.float64)  # Q_hat

    def _select_arms(self) -> np.ndarray:
        q = self.arm_value_estimates * self.theta
        g = np.random.gumbel(loc=0.0, scale=1.0, size=(self.n_bandits, self.n_arms))
        return np.argmax(q + g, axis=1).astype(np.int32)

    def _update_estimates(self, chosen_arms: np.ndarray, rewards: np.ndarray) -> None:
        rows = np.arange(self.n_bandits)
        self.arm_pull_counts[rows, chosen_arms] += 1
        pulls = self.arm_pull_counts[rows, chosen_arms].astype(np.float64)

        q_old = self.arm_value_estimates[rows, chosen_arms]
        self.arm_value_estimates[rows, chosen_arms] = q_old + (rewards - q_old) / pulls

    def step(self):
        chosen_arms = self._select_arms()
        rewards = self.bandit.bulk_pull(chosen_arms)

        self._update_estimates(chosen_arms, rewards)
        self.total_steps += 1

        return chosen_arms, rewards


# [LEGACY]
class BoltzmannArbitraryNoiseBulkAlgorithm:
    """
    Bulk Boltzmann-like exploration by adding i.i.d. noise Z_a from an arbitrary distribution:
        A = argmax_a { theta * Q_hat[a] + Z_a }

    Supports at least: "gumbel", "cauchy", "beta", "betaprime", "chi".
    For "beta", parameters (a, b) can be provided.
    For "betaprime", parameters (a, b) can be provided (requires scipy).
    For "chi", parameter (df) can be provided (requires scipy).

    Notes:
    - For scipy-backed noises, sampling is done per-step and then reshaped to (B, K).
      This is still vectorized (one scipy call per step), avoiding Python loops over bandits.
    """

    def __init__(
        self,
        bandit: Gang_of_Bandits,
        theta: float,
        noise: str = "gumbel",
        noise_params: dict | None = None,
    ):
        self.bandit = bandit
        self.theta = float(theta)

        self.noise = str(noise).lower().strip()
        self.noise_params = noise_params or {}

        self.n_bandits = bandit.n_bandits
        self.n_arms = bandit.n_arms

        self.total_steps = 0

        self.arm_pull_counts = np.zeros((self.n_bandits, self.n_arms), dtype=np.int32)
        self.arm_value_estimates = np.zeros((self.n_bandits, self.n_arms), dtype=np.float64)  # Q_hat

    def _sample_noise_matrix(self) -> np.ndarray:
        n = self.n_bandits * self.n_arms

        if self.noise == "gumbel":
            loc = float(self.noise_params.get("loc", 0.0))
            scale = float(self.noise_params.get("scale", 1.0))
            z = np.random.gumbel(loc=loc, scale=scale, size=n)
            return z.reshape(self.n_bandits, self.n_arms)

        if self.noise == "cauchy":
            loc = float(self.noise_params.get("loc", 0.0))
            scale = float(self.noise_params.get("scale", 1.0))
            z = loc + scale * np.random.standard_cauchy(size=n)
            return z.reshape(self.n_bandits, self.n_arms)

        if self.noise == "beta":
            a = float(self.noise_params.get("a", 1.0))
            b = float(self.noise_params.get("b", 1.0))
            z = np.random.beta(a, b, size=n)
            return z.reshape(self.n_bandits, self.n_arms)

        if self.noise == "betaprime":
            if _scipy_stats is None:
                raise ImportError("scipy is required for betaprime noise (scipy.stats.betaprime).")
            a = float(self.noise_params.get("a", 1.0))
            b = float(self.noise_params.get("b", 1.0))
            z = _scipy_stats.betaprime(a, b).rvs(size=n, random_state=None)
            return np.asarray(z, dtype=float).reshape(self.n_bandits, self.n_arms)

        if self.noise == "chi":
            if _scipy_stats is None:
                raise ImportError("scipy is required for chi noise (scipy.stats.chi).")
            df = float(self.noise_params.get("df", 1.0))
            z = _scipy_stats.chi(df).rvs(size=n, random_state=None)
            return np.asarray(z, dtype=float).reshape(self.n_bandits, self.n_arms)

        # Allow passing a scipy-like frozen distribution directly
        dist = self.noise_params.get("dist", None)
        if dist is not None and hasattr(dist, "rvs"):
            z = dist.rvs(size=n)
            return np.asarray(z, dtype=float).reshape(self.n_bandits, self.n_arms)

        raise ValueError(f"Unknown noise distribution '{self.noise}'")

    def _select_arms(self) -> np.ndarray:
        q = self.arm_value_estimates * self.theta
        z = self._sample_noise_matrix()
        return np.argmax(q + z, axis=1).astype(np.int32)

    def _update_estimates(self, chosen_arms: np.ndarray, rewards: np.ndarray) -> None:
        rows = np.arange(self.n_bandits)
        self.arm_pull_counts[rows, chosen_arms] += 1
        pulls = self.arm_pull_counts[rows, chosen_arms].astype(np.float64)

        q_old = self.arm_value_estimates[rows, chosen_arms]
        self.arm_value_estimates[rows, chosen_arms] = q_old + (rewards - q_old) / pulls

    def step(self):
        chosen_arms = self._select_arms()
        rewards = self.bandit.bulk_pull(chosen_arms)

        self._update_estimates(chosen_arms, rewards)
        self.total_steps += 1

        return chosen_arms, rewards


# [LEGACY]
class GumbelScaledBonusBulkAlgorithm:
    """
    Bulk version of:
        A_t ∈ argmax_a { Q_hat[a] + sqrt(C / T_a) * Z_a },
    with Z_a i.i.d. standard Gumbel.

    For T_a = 0, we force exploration by returning +inf (so each arm is tried at least once).
    If C < 0, the sqrt is not real; we treat C_eff = 0 (no bonus).
    """

    def __init__(self, bandit: Gang_of_Bandits, C: float):
        self.bandit = bandit
        self.C = float(C)

        self.n_bandits = bandit.n_bandits
        self.n_arms = bandit.n_arms

        self.total_steps = 0

        self.arm_pull_counts = np.zeros((self.n_bandits, self.n_arms), dtype=np.int32)
        self.arm_value_estimates = np.zeros((self.n_bandits, self.n_arms), dtype=np.float64)  # Q_hat

    def _select_arms(self) -> np.ndarray:
        c_eff = self.C if self.C >= 0.0 else 0.0

        pulls = self.arm_pull_counts.astype(np.float64)  # (B, K)
        q_hat = self.arm_value_estimates                 # (B, K)

        z = np.random.gumbel(loc=0.0, scale=1.0, size=(self.n_bandits, self.n_arms))

        scores = np.empty_like(q_hat)

        mask = pulls > 0.0
        # where pulls>0: q + sqrt(C/pulls)*z
        scores[mask] = q_hat[mask] + np.sqrt(c_eff / pulls[mask]) * z[mask]
        # where pulls==0: +inf to force trying every arm
        scores[~mask] = np.inf

        return np.argmax(scores, axis=1).astype(np.int32)

    def _update_estimates(self, chosen_arms: np.ndarray, rewards: np.ndarray) -> None:
        rows = np.arange(self.n_bandits)
        self.arm_pull_counts[rows, chosen_arms] += 1
        pulls = self.arm_pull_counts[rows, chosen_arms].astype(np.float64)

        q_old = self.arm_value_estimates[rows, chosen_arms]
        self.arm_value_estimates[rows, chosen_arms] = q_old + (rewards - q_old) / pulls

    def step(self):
        chosen_arms = self._select_arms()
        rewards = self.bandit.bulk_pull(chosen_arms)

        self._update_estimates(chosen_arms, rewards)
        self.total_steps += 1

        return chosen_arms, rewards





# [LEGACY]
class BoltzmannExplorationAlgorithm:
    """
    Simple Boltzmann (softmax) exploration:
        P(A=a) ∝ exp(theta * Q_hat[a])
    """

    def __init__(self, bandit: Bandit, theta: float):
        """
        Initialize Boltzmann exploration.

        Parameters
        ----------
        bandit : Bandit
            The bandit environment.
        theta : float
            Inverse-temperature parameter (larger -> greedier).
        """
        self.bandit = bandit
        self.theta = float(theta)

        # Number of arms
        self.n_arms = bandit.n_arms

        # Tracking variables
        self.total_steps = 0
        self.arm_pull_counts = [0 for _ in range(self.n_arms)]
        self.arm_value_estimates = [0.0 for _ in range(self.n_arms)]  # Q_hat


    def _softmax_probs(self):
        # Numerically-stable softmax over theta * Q_hat
        logits = np.array(self.arm_value_estimates, dtype=float) * self.theta
        logits = logits - np.max(logits)
        exp_logits = np.exp(logits)
        s = float(np.sum(exp_logits))
        if s <= 0.0 or not np.isfinite(s):
            # Fallback to uniform (should be rare)
            return np.ones(self.n_arms, dtype=float) / float(self.n_arms)
        return exp_logits / s

    def _select_arm(self) -> int:
        probs = self._softmax_probs()
        return int(np.random.choice(self.n_arms, p=probs))

    def _update_estimate(self, chosen_arm: int, reward: float):
        self.arm_pull_counts[chosen_arm] += 1
        pulls = self.arm_pull_counts[chosen_arm]
        q_old = self.arm_value_estimates[chosen_arm]
        self.arm_value_estimates[chosen_arm] = q_old + (1.0 / pulls) * (reward - q_old)

    def step(self):
        """
        Perform ONE step of Boltzmann exploration.

        Returns
        -------
        chosen_arm : int
            Index of the arm pulled.
        reward : float
            Reward obtained from the bandit.
        """
        chosen_arm = self._select_arm()

        reward = self.bandit.pull(chosen_arm)

        self._update_estimate(chosen_arm, reward)
        self.total_steps += 1

        return chosen_arm, reward


# [LEGACY]
class BoltzmannGumbelTrickAlgorithm:
    """
    Boltzmann exploration via the Gumbel trick:
        A = argmax_a { theta * Q_hat[a] + G_a },  G_a i.i.d. standard Gumbel
    """

    def __init__(self, bandit: Bandit, theta: float):
        """
        Initialize Boltzmann exploration via the Gumbel trick.

        Parameters
        ----------
        bandit : Bandit
            The bandit environment.
        theta : float
            Inverse-temperature parameter.
        """
        self.bandit = bandit
        self.theta = float(theta)

        self.n_arms = bandit.n_arms

        self.total_steps = 0
        self.arm_pull_counts = [0 for _ in range(self.n_arms)]
        self.arm_value_estimates = [0.0 for _ in range(self.n_arms)]  # Q_hat


    def _select_arm(self) -> int:
        q = np.array(self.arm_value_estimates, dtype=float) * self.theta
        g = np.random.gumbel(loc=0.0, scale=1.0, size=self.n_arms)
        scores = q + g
        return int(np.argmax(scores))

    def _update_estimate(self, chosen_arm: int, reward: float):
        self.arm_pull_counts[chosen_arm] += 1
        pulls = self.arm_pull_counts[chosen_arm]
        q_old = self.arm_value_estimates[chosen_arm]
        self.arm_value_estimates[chosen_arm] = q_old + (1.0 / pulls) * (reward - q_old)

    def step(self):
        """
        Perform ONE step of Boltzmann (Gumbel trick).

        Returns
        -------
        chosen_arm : int
            Index of the arm pulled.
        reward : float
            Reward obtained from the bandit.
        """
        chosen_arm = self._select_arm()

        reward = self.bandit.pull(chosen_arm)

        self._update_estimate(chosen_arm, reward)
        self.total_steps += 1

        return chosen_arm, reward


# [LEGACY]
class BoltzmannArbitraryNoiseAlgorithm:
    """
    Boltzmann-like exploration by adding i.i.d. noise Z_a from an arbitrary distribution:
        A = argmax_a { theta * Q_hat[a] + Z_a }

    Supports at least: "gumbel", "cauchy", "beta", "betaprime", "chi".
    For "beta", parameters (a, b) can be provided.
    For "betaprime", parameters (a, b) can be provided (requires scipy).
    For "chi", parameter (df) can be provided (requires scipy).
    """

    def __init__(self, bandit: Bandit, theta: float, noise: str = "gumbel", noise_params: dict | None = None):
        """
        Initialize arbitrary-noise Boltzmann exploration.

        Parameters
        ----------
        bandit : Bandit
            The bandit environment.
        theta : float
            Inverse-temperature parameter.
        noise : str
            Noise distribution name.
        noise_params : dict | None
            Optional distribution parameters.
        """
        self.bandit = bandit
        self.theta = float(theta)

        self.noise = str(noise).lower().strip()
        self.noise_params = noise_params or {}

        self.n_arms = bandit.n_arms

        self.total_steps = 0
        self.arm_pull_counts = [0 for _ in range(self.n_arms)]
        self.arm_value_estimates = [0.0 for _ in range(self.n_arms)]  # Q_hat


    def _sample_noise(self, size: int):
        n = int(size)

        if self.noise == "gumbel":
            loc = float(self.noise_params.get("loc", 0.0))
            scale = float(self.noise_params.get("scale", 1.0))
            return np.random.gumbel(loc=loc, scale=scale, size=n)

        if self.noise == "cauchy":
            loc = float(self.noise_params.get("loc", 0.0))
            scale = float(self.noise_params.get("scale", 1.0))
            return loc + scale * np.random.standard_cauchy(size=n)

        if self.noise == "beta":
            a = float(self.noise_params.get("a", 1.0))
            b = float(self.noise_params.get("b", 1.0))
            return np.random.beta(a, b, size=n)

        if self.noise == "betaprime":
            if _scipy_stats is None:
                raise ImportError("scipy is required for betaprime noise (scipy.stats.betaprime).")
            a = float(self.noise_params.get("a", 1.0))
            b = float(self.noise_params.get("b", 1.0))
            return _scipy_stats.betaprime(a, b).rvs(size=n, random_state=None)

        if self.noise == "chi":
            if _scipy_stats is None:
                raise ImportError("scipy is required for chi noise (scipy.stats.chi).")
            df = float(self.noise_params.get("df", 1.0))
            return _scipy_stats.chi(df).rvs(size=n, random_state=None)

        # Allow passing a scipy-like frozen distribution directly
        dist = self.noise_params.get("dist", None)
        if dist is not None and hasattr(dist, "rvs"):
            return dist.rvs(size=n)

        raise ValueError(f"Unknown noise distribution '{self.noise}'")

    def _select_arm(self) -> int:
        q = np.array(self.arm_value_estimates, dtype=float) * self.theta
        z = np.array(self._sample_noise(self.n_arms), dtype=float)
        scores = q + z
        return int(np.argmax(scores))

    def _update_estimate(self, chosen_arm: int, reward: float):
        self.arm_pull_counts[chosen_arm] += 1
        pulls = self.arm_pull_counts[chosen_arm]
        q_old = self.arm_value_estimates[chosen_arm]
        self.arm_value_estimates[chosen_arm] = q_old + (1.0 / pulls) * (reward - q_old)

    def step(self):
        """
        Perform ONE step of arbitrary-noise Boltzmann exploration.

        Returns
        -------
        chosen_arm : int
            Index of the arm pulled.
        reward : float
            Reward obtained from the bandit.
        """
        chosen_arm = self._select_arm()

        reward = self.bandit.pull(chosen_arm)

        self._update_estimate(chosen_arm, reward)
        self.total_steps += 1

        return chosen_arm, reward


# [LEGACY]
class GumbelScaledBonusAlgorithm:
    """
    Version where:
        A_t ∈ argmax_a { Q_hat[a] + sqrt(C / T_a) * Z_a },
    with Z_a i.i.d. standard Gumbel.

    For T_a = 0, we force exploration by returning +inf (so each arm is tried at least once).
    If C < 0, the sqrt is not real; we treat C_eff = 0 (no bonus).
    """

    def __init__(self, bandit: Bandit, C: float):
        """
        Initialize the scaled-gumbel-bonus algorithm.

        Parameters
        ----------
        bandit : Bandit
            The bandit environment.
        C : float
            Bonus scaling parameter (intended >= 0). If < 0, treated as 0.
        """
        self.bandit = bandit
        self.C = float(C)

        self.n_arms = bandit.n_arms

        self.total_steps = 0
        self.arm_pull_counts = [0 for _ in range(self.n_arms)]
        self.arm_value_estimates = [0.0 for _ in range(self.n_arms)]  # Q_hat


    def _select_arm(self) -> int:
        c_eff = self.C if self.C >= 0.0 else 0.0

        scores = []
        z = np.random.gumbel(loc=0.0, scale=1.0, size=self.n_arms)

        for a in range(self.n_arms):
            pulls = self.arm_pull_counts[a]
            if pulls == 0:
                scores.append(float("inf"))
            else:
                bonus_scale = np.sqrt(c_eff / float(pulls))
                scores.append(float(self.arm_value_estimates[a] + bonus_scale * z[a]))

        return int(scores.index(max(scores)))

    def _update_estimate(self, chosen_arm: int, reward: float):
        self.arm_pull_counts[chosen_arm] += 1
        pulls = self.arm_pull_counts[chosen_arm]
        q_old = self.arm_value_estimates[chosen_arm]
        self.arm_value_estimates[chosen_arm] = q_old + (1.0 / pulls) * (reward - q_old)

    def step(self):
        """
        Perform ONE step of the scaled-gumbel-bonus algorithm.

        Returns
        -------
        chosen_arm : int
            Index of the arm pulled.
        reward : float
            Reward obtained from the bandit.
        """
        chosen_arm = self._select_arm()

        reward = self.bandit.pull(chosen_arm)

        self._update_estimate(chosen_arm, reward)
        self.total_steps += 1

        return chosen_arm, reward
    
    # src/boltzmann_bulk.py


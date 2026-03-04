# src/ucb.py

from __future__ import annotations
from src.bandits import Bandit, Gang_of_Bandits
import numpy as np

# from util.io_helpers import log, out
# do not let classes log themselves.


class UCBAlgorithm:
    """
    UCB as typically presented for bounded rewards (e.g., [0,1]) in many lectures:
        UCB_a(t) = Q_hat[a] + sqrt( (2 * ln(1/delta)) / T_a )

    For T_a = 0, we force exploration by returning +inf.
    """

    def __init__(self, bandit: Bandit, delta: float):
        """
        Initialize the UCB algorithm.

        Parameters
        ----------
        bandit : Bandit
            The bandit environment.
        delta : float
            Confidence parameter in (0, 1).
        """
        self.bandit = bandit
        self.delta = float(delta)

        # Number of arms
        self.n_arms = bandit.n_arms

        # Tracking variables
        self.total_steps = 0
        self.arm_pull_counts = [0 for _ in range(self.n_arms)]
        self.arm_value_estimates = [0.0 for _ in range(self.n_arms)]  # Q_hat


    def _ucb_value(self, arm_index: int) -> float:
        # Compute UCB_a(t, delta) using counts up to time t (self.total_steps).
        pulls = self.arm_pull_counts[arm_index]

        # Force each arm to be tried at least once
        if pulls == 0:
            return float("inf")

        q_hat = self.arm_value_estimates[arm_index]
        bonus = np.sqrt((2.0 * np.log(1.0 / self.delta)) / pulls)
        return float(q_hat + bonus)

    def _select_arm(self) -> int:
        # Select argmax_a UCB_a (ties broken by first occurrence).
        ucb_values = []
        for a in range(self.n_arms):
            ucb_values.append(self._ucb_value(a))
        return int(ucb_values.index(max(ucb_values)))

    def _update_estimate(self, chosen_arm: int, reward: float):
        # Incremental mean update:
        # Q_hat[a] <- Q_hat[a] + (1/T_a) * (reward - Q_hat[a])
        self.arm_pull_counts[chosen_arm] += 1
        pulls = self.arm_pull_counts[chosen_arm]
        q_old = self.arm_value_estimates[chosen_arm]
        self.arm_value_estimates[chosen_arm] = q_old + (1.0 / pulls) * (reward - q_old)

    def step(self):
        """
        Perform ONE step of the UCB algorithm.

        Returns
        -------
        chosen_arm : int
            Index of the arm pulled.
        reward : float
            Reward obtained from the bandit.
        """
        chosen_arm = self._select_arm()

        # Pull the arm
        reward = self.bandit.pull(chosen_arm)

        # Update tracking statistics
        self._update_estimate(chosen_arm, reward)
        self.total_steps += 1

        return chosen_arm, reward


class UCBSubGaussianAlgorithm:
    """
    UCB adapted to sigma-subgaussian rewards:
        UCB_a(t) = Q_hat[a] + sigma * sqrt( (2 * ln(1/delta)) / T_a )

    For T_a = 0, we force exploration by returning +inf.
    """

    def __init__(self, bandit: Bandit, delta: float, sigma: float):
        """
        Initialize the sigma-subgaussian UCB algorithm.

        Parameters
        ----------
        bandit : Bandit
            The bandit environment.
        delta : float
            Confidence parameter in (0, 1).
        sigma : float
            Subgaussian parameter (>= 0).
        """
        self.bandit = bandit
        self.delta = float(delta)
        self.sigma = float(sigma)

        # Number of arms
        self.n_arms = bandit.n_arms

        # Tracking variables
        self.total_steps = 0
        self.arm_pull_counts = [0 for _ in range(self.n_arms)]
        self.arm_value_estimates = [0.0 for _ in range(self.n_arms)]  # Q_hat


    def _ucb_value(self, arm_index: int) -> float:
        pulls = self.arm_pull_counts[arm_index]

        # Force each arm to be tried at least once
        if pulls == 0:
            return float("inf")

        q_hat = self.arm_value_estimates[arm_index]
        bonus = self.sigma * np.sqrt((2.0 * np.log(1.0 / self.delta)) / pulls)
        return float(q_hat + bonus)

    def _select_arm(self) -> int:
        ucb_values = []
        for a in range(self.n_arms):
            ucb_values.append(self._ucb_value(a))
        return int(ucb_values.index(max(ucb_values)))

    def _update_estimate(self, chosen_arm: int, reward: float):
        self.arm_pull_counts[chosen_arm] += 1
        pulls = self.arm_pull_counts[chosen_arm]
        q_old = self.arm_value_estimates[chosen_arm]
        self.arm_value_estimates[chosen_arm] = q_old + (1.0 / pulls) * (reward - q_old)

    def step(self):
        """
        Perform ONE step of the sigma-subgaussian UCB algorithm.

        Returns
        -------
        chosen_arm : int
            Index of the arm pulled.
        reward : float
            Reward obtained from the bandit.
        """
        chosen_arm = self._select_arm()

        # Pull the arm
        reward = self.bandit.pull(chosen_arm)

        # Update tracking statistics
        self._update_estimate(chosen_arm, reward)
        self.total_steps += 1

        return chosen_arm, reward
    

class UCBulkAlgorithm:
    """
    Bulk UCB for bounded rewards (e.g., [0,1]):

        UCB_a(t) = Q_hat[a] + sqrt( (2 * ln(1/delta)) / T_a )

    For T_a = 0, we force exploration by setting UCB to +inf.
    Runs over many bandits at once (Gang_of_Bandits).
    """

    def __init__(self, bandit: Gang_of_Bandits, delta: float):
        self.bandit = bandit
        self.delta = float(delta)

        self.n_bandits = bandit.n_bandits
        self.n_arms = bandit.n_arms

        self.total_steps = 0

        # Vectorized tracking state
        self.arm_pull_counts = np.zeros((self.n_bandits, self.n_arms), dtype=np.int32)
        self.arm_value_estimates = np.zeros((self.n_bandits, self.n_arms), dtype=np.float64)  # Q_hat

        # Precompute constant factor
        self._log_term = float(np.log(1.0 / self.delta))

    def _select_arms(self) -> np.ndarray:
        # Compute UCB values for all bandits/arms.
        pulls = self.arm_pull_counts.astype(np.float64)  # (B, K)
        q_hat = self.arm_value_estimates                 # (B, K)

        # bonus = sqrt((2*log(1/delta)) / pulls) for pulls>0, else inf
        ucb = np.empty_like(q_hat)

        mask = pulls > 0.0
        ucb[mask] = q_hat[mask] + np.sqrt((2.0 * self._log_term) / pulls[mask])
        ucb[~mask] = np.inf

        # Per-bandit argmax (ties go to first occurrence, like Python list.index(max(...))).
        return np.argmax(ucb, axis=1).astype(np.int32)

    def _update_estimates(self, chosen_arms: np.ndarray, rewards: np.ndarray) -> None:
        rows = np.arange(self.n_bandits)

        # Increment counts
        self.arm_pull_counts[rows, chosen_arms] += 1
        pulls = self.arm_pull_counts[rows, chosen_arms].astype(np.float64)

        # Incremental mean update:
        # Q <- Q + (1/pulls) * (reward - Q)
        q_old = self.arm_value_estimates[rows, chosen_arms]
        self.arm_value_estimates[rows, chosen_arms] = q_old + (rewards - q_old) / pulls

    def step(self):
        """
        Perform ONE synchronized bulk step.

        Returns
        -------
        chosen_arms : np.ndarray, shape (n_bandits,)
        rewards : np.ndarray, shape (n_bandits,)
        """
        chosen_arms = self._select_arms()
        rewards = self.bandit.bulk_pull(chosen_arms)

        self._update_estimates(chosen_arms, rewards)
        self.total_steps += 1

        return chosen_arms, rewards


class UCBSubGaussianBulkAlgorithm:
    """
    Bulk UCB for sigma-subgaussian rewards:

        UCB_a(t) = Q_hat[a] + sigma * sqrt( (2 * ln(1/delta)) / T_a )

    For T_a = 0, we force exploration by setting UCB to +inf.
    """

    def __init__(self, bandit: Gang_of_Bandits, delta: float, sigma: float):
        self.bandit = bandit
        self.delta = float(delta)
        self.sigma = float(sigma)

        self.n_bandits = bandit.n_bandits
        self.n_arms = bandit.n_arms

        self.total_steps = 0

        self.arm_pull_counts = np.zeros((self.n_bandits, self.n_arms), dtype=np.int32)
        self.arm_value_estimates = np.zeros((self.n_bandits, self.n_arms), dtype=np.float64)  # Q_hat

        self._log_term = float(np.log(1.0 / self.delta))

    def _select_arms(self) -> np.ndarray:
        pulls = self.arm_pull_counts.astype(np.float64)  # (B, K)
        q_hat = self.arm_value_estimates                 # (B, K)

        ucb = np.empty_like(q_hat)

        mask = pulls > 0.0
        ucb[mask] = q_hat[mask] + self.sigma * np.sqrt((2.0 * self._log_term) / pulls[mask])
        ucb[~mask] = np.inf

        return np.argmax(ucb, axis=1).astype(np.int32)

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
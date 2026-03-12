# src/greedy.py
# [CORE] GreedyBulkAlgorithm, EpsilonGreedyFixedBulkAlgorithm, EpsilonGreedyDecreasingBulkAlgorithm, 
# [LEGACY] GreedyAlgorithm, EpsilonGreedyFixedAlgorithm, EpsilonGreedyDecreasingAlgorithm

from __future__ import annotations
from src.bandits import Bandit, Gang_of_Bandits
import numpy as np

# from util.io_helpers import log, out
# do not let classes log themselves.


class GreedyBulkAlgorithm:
    """
    Purely greedy (no exploration), but run in bulk over many bandits at once:
    always pull argmax_a Q_hat[b, a] for each bandit b.
    """

    def __init__(self, bandit: Gang_of_Bandits):
        self.bandit = bandit

        self.n_bandits = bandit.n_bandits
        self.n_arms = bandit.n_arms

        self.total_steps = 0

        # Vectorized tracking state
        self.arm_pull_counts = np.zeros((self.n_bandits, self.n_arms), dtype=np.int32)
        self.arm_value_estimates = np.zeros((self.n_bandits, self.n_arms), dtype=np.float64)  # Q_hat

    def _select_greedy_arms(self) -> np.ndarray:
        # Per-bandit argmax over arms.
        return np.argmax(self.arm_value_estimates, axis=1).astype(np.int32)

    def _update_estimates(self, chosen_arms: np.ndarray, rewards: np.ndarray) -> None:
        rows = np.arange(self.n_bandits)

        # Increment counts for chosen arms
        self.arm_pull_counts[rows, chosen_arms] += 1
        pulls = self.arm_pull_counts[rows, chosen_arms].astype(np.float64)

        # Incremental mean update on just the selected arms:
        # Q <- Q + (1/pulls) * (reward - Q)
        q_old = self.arm_value_estimates[rows, chosen_arms]
        self.arm_value_estimates[rows, chosen_arms] = q_old + (rewards - q_old) / pulls

    def step(self):
        """
        Perform ONE synchronized step over all bandits.

        Returns
        -------
        chosen_arms : np.ndarray, shape (n_bandits,)
        rewards : np.ndarray, shape (n_bandits,)
        """
        chosen_arms = self._select_greedy_arms()
        rewards = self.bandit.bulk_pull(chosen_arms)

        self._update_estimates(chosen_arms, rewards)
        self.total_steps += 1

        return chosen_arms, rewards


class EpsilonGreedyFixedBulkAlgorithm:
    """
    Epsilon-greedy with fixed epsilon in (0,1), run in bulk over many bandits.
    """

    def __init__(self, bandit: Gang_of_Bandits, epsilon: float):
        self.bandit = bandit
        self.epsilon = float(epsilon)

        self.n_bandits = bandit.n_bandits
        self.n_arms = bandit.n_arms

        self.total_steps = 0

        self.arm_pull_counts = np.zeros((self.n_bandits, self.n_arms), dtype=np.int32)
        self.arm_value_estimates = np.zeros((self.n_bandits, self.n_arms), dtype=np.float64)  # Q_hat

    def _select_greedy_arms(self) -> np.ndarray:
        return np.argmax(self.arm_value_estimates, axis=1).astype(np.int32)

    def _select_random_arms(self) -> np.ndarray:
        return np.random.randint(0, self.n_arms, size=self.n_bandits, dtype=np.int32)

    def _update_estimates(self, chosen_arms: np.ndarray, rewards: np.ndarray) -> None:
        rows = np.arange(self.n_bandits)
        self.arm_pull_counts[rows, chosen_arms] += 1
        pulls = self.arm_pull_counts[rows, chosen_arms].astype(np.float64)

        q_old = self.arm_value_estimates[rows, chosen_arms]
        self.arm_value_estimates[rows, chosen_arms] = q_old + (rewards - q_old) / pulls

    def step(self):
        """
        Perform ONE synchronized step over all bandits.

        Returns
        -------
        chosen_arms : np.ndarray, shape (n_bandits,)
        rewards : np.ndarray, shape (n_bandits,)
        """
        u = np.random.rand(self.n_bandits)
        explore = u < self.epsilon # this is an array, as of course not all bandits explore in the same rounds.

        greedy_arms = self._select_greedy_arms()
        random_arms = self._select_random_arms()

        chosen_arms = np.where(explore, random_arms, greedy_arms).astype(np.int32)
        rewards = self.bandit.bulk_pull(chosen_arms)

        self._update_estimates(chosen_arms, rewards)
        self.total_steps += 1

        return chosen_arms, rewards


class EpsilonGreedyDecreasingBulkAlgorithm:
    """
    Epsilon-greedy with decreasing epsilon_t, run in bulk over many bandits.

    epsilon_t = min(1, epsilon0/(t+1)), where t = total_steps (starting at 0).
    """

    def __init__(self, bandit: Gang_of_Bandits, epsilon0: float):
        self.bandit = bandit
        self.epsilon0 = float(epsilon0)

        self.n_bandits = bandit.n_bandits
        self.n_arms = bandit.n_arms

        self.total_steps = 0

        self.arm_pull_counts = np.zeros((self.n_bandits, self.n_arms), dtype=np.int32)
        self.arm_value_estimates = np.zeros((self.n_bandits, self.n_arms), dtype=np.float64)  # Q_hat

    def _epsilon_t(self) -> float:
        if self.epsilon0 <= 0.0:
            return 0.0
        return min(1.0, self.epsilon0 / (self.total_steps + 1.0))

    def _select_greedy_arms(self) -> np.ndarray:
        return np.argmax(self.arm_value_estimates, axis=1).astype(np.int32)

    def _select_random_arms(self) -> np.ndarray:
        return np.random.randint(0, self.n_arms, size=self.n_bandits, dtype=np.int32)

    def _update_estimates(self, chosen_arms: np.ndarray, rewards: np.ndarray) -> None:
        rows = np.arange(self.n_bandits)
        self.arm_pull_counts[rows, chosen_arms] += 1
        pulls = self.arm_pull_counts[rows, chosen_arms].astype(np.float64)

        q_old = self.arm_value_estimates[rows, chosen_arms]
        self.arm_value_estimates[rows, chosen_arms] = q_old + (rewards - q_old) / pulls

    def step(self):
        """
        Perform ONE synchronized step over all bandits.

        Returns
        -------
        chosen_arms : np.ndarray, shape (n_bandits,)
        rewards : np.ndarray, shape (n_bandits,)
        """
        eps = self._epsilon_t()
        u = np.random.rand(self.n_bandits)
        explore = u < eps

        greedy_arms = self._select_greedy_arms()
        random_arms = self._select_random_arms()

        chosen_arms = np.where(explore, random_arms, greedy_arms).astype(np.int32)
        rewards = self.bandit.bulk_pull(chosen_arms)

        self._update_estimates(chosen_arms, rewards)
        self.total_steps += 1

        return chosen_arms, rewards



# [LEGACY]
class GreedyAlgorithm:
    """
    Purely greedy (no exploration): always pull argmax_a Q_hat[a].
    """

    def __init__(self, bandit: Bandit):
        """
        Initialize the Greedy algorithm.

        Parameters
        ----------
        bandit : Bandit
            The bandit environment.
        """
        self.bandit = bandit

        # Number of arms
        self.n_arms = bandit.n_arms

        # Tracking variables
        self.total_steps = 0
        self.arm_pull_counts = [0 for _ in range(self.n_arms)]
        self.arm_value_estimates = [0.0 for _ in range(self.n_arms)]  # Q_hat


    def _select_greedy_arm(self) -> int:
        # Select argmax_a Q_hat[a] (ties broken by first occurrence).
        return self.arm_value_estimates.index(max(self.arm_value_estimates))

    def _update_estimate(self, chosen_arm: int, reward: float):
        # Incremental mean update:
        # Q_hat[a] <- Q_hat[a] + (1/T_a) * (reward - Q_hat[a])
        self.arm_pull_counts[chosen_arm] += 1
        pulls = self.arm_pull_counts[chosen_arm]
        q_old = self.arm_value_estimates[chosen_arm]
        self.arm_value_estimates[chosen_arm] = q_old + (1.0 / pulls) * (reward - q_old)

    def step(self):
        """
        Perform ONE step of the Greedy algorithm.

        Returns
        -------
        chosen_arm : int
            Index of the arm pulled.
        reward : float
            Reward obtained from the bandit.
        """
        chosen_arm = self._select_greedy_arm()

        # Pull the arm
        reward = self.bandit.pull(chosen_arm)

        # Update tracking statistics
        self._update_estimate(chosen_arm, reward)
        self.total_steps += 1

        return chosen_arm, reward

# [LEGACY]
class EpsilonGreedyFixedAlgorithm:
    """
    Epsilon-greedy with a fixed exploration rate epsilon in (0,1).
    """

    def __init__(self, bandit: Bandit, epsilon: float):
        """
        Initialize the fixed-epsilon Greedy algorithm.

        Parameters
        ----------
        bandit : Bandit
            The bandit environment.
        epsilon : float
            Fixed exploration rate in (0, 1).
        """
        self.bandit = bandit
        self.epsilon = float(epsilon)

        # Number of arms
        self.n_arms = bandit.n_arms

        # Tracking variables
        self.total_steps = 0
        self.arm_pull_counts = [0 for _ in range(self.n_arms)]
        self.arm_value_estimates = [0.0 for _ in range(self.n_arms)]  # Q_hat


    def _select_greedy_arm(self) -> int:
        # Select argmax_a Q_hat[a] (ties broken by first occurrence).
        return self.arm_value_estimates.index(max(self.arm_value_estimates))

    def _select_random_arm(self) -> int:
        # Uniformly choose an arm.
        return int(np.random.randint(0, self.n_arms))

    def _update_estimate(self, chosen_arm: int, reward: float):
        # Incremental mean update.
        self.arm_pull_counts[chosen_arm] += 1
        pulls = self.arm_pull_counts[chosen_arm]
        q_old = self.arm_value_estimates[chosen_arm]
        self.arm_value_estimates[chosen_arm] = q_old + (1.0 / pulls) * (reward - q_old)

    def step(self):
        """
        Perform ONE step of the fixed-epsilon Greedy algorithm.

        Returns
        -------
        chosen_arm : int
            Index of the arm pulled.
        reward : float
            Reward obtained from the bandit.
        """
        u = float(np.random.rand())

        # Decide which arm to pull
        if u < self.epsilon:
            chosen_arm = self._select_random_arm()
        else:
            chosen_arm = self._select_greedy_arm()

        # Pull the arm
        reward = self.bandit.pull(chosen_arm)

        # Update tracking statistics
        self._update_estimate(chosen_arm, reward)
        self.total_steps += 1

        return chosen_arm, reward

# [LEGACY]
class EpsilonGreedyDecreasingAlgorithm:
    """
    Epsilon-greedy with a decreasing exploration rate epsilon_t.

    Here we use: epsilon_t = min(1, epsilon0 / (t+1)),
    where t = total_steps (starting at 0).
    """

    def __init__(self, bandit: Bandit, epsilon0: float):
        """
        Initialize the decreasing-epsilon Greedy algorithm.

        Parameters
        ----------
        bandit : Bandit
            The bandit environment.
        epsilon0 : float
            Initial exploration scale (> 0). The used rate is epsilon_t = min(1, epsilon0/(t+1)).
        """
        self.bandit = bandit
        self.epsilon0 = float(epsilon0)

        # Number of arms
        self.n_arms = bandit.n_arms

        # Tracking variables
        self.total_steps = 0
        self.arm_pull_counts = [0 for _ in range(self.n_arms)]
        self.arm_value_estimates = [0.0 for _ in range(self.n_arms)]  # Q_hat


    def _epsilon_t(self) -> float:
        # Decreasing exploration rate.
        t = self.total_steps  # starts at 0
        if self.epsilon0 <= 0.0:
            return 0.0
        return min(1.0, self.epsilon0 / (t + 1.0))

    def _select_greedy_arm(self) -> int:
        # Select argmax_a Q_hat[a] (ties broken by first occurrence).
        return self.arm_value_estimates.index(max(self.arm_value_estimates))

    def _select_random_arm(self) -> int:
        # Uniformly choose an arm.
        return int(np.random.randint(0, self.n_arms))

    def _update_estimate(self, chosen_arm: int, reward: float):
        # Incremental mean update.
        self.arm_pull_counts[chosen_arm] += 1
        pulls = self.arm_pull_counts[chosen_arm]
        q_old = self.arm_value_estimates[chosen_arm]
        self.arm_value_estimates[chosen_arm] = q_old + (1.0 / pulls) * (reward - q_old)

    def step(self):
        """
        Perform ONE step of the decreasing-epsilon Greedy algorithm.

        Returns
        -------
        chosen_arm : int
            Index of the arm pulled.
        reward : float
            Reward obtained from the bandit.
        """
        eps = self._epsilon_t()
        u = float(np.random.rand())

        # Decide which arm to pull
        if u < eps:
            chosen_arm = self._select_random_arm()
        else:
            chosen_arm = self._select_greedy_arm()

        # Pull the arm
        reward = self.bandit.pull(chosen_arm)

        # Update tracking statistics
        self._update_estimate(chosen_arm, reward)
        self.total_steps += 1

        return chosen_arm, reward

# src/policy_gradient.py

from __future__ import annotations
from src.bandits import Bandit, Gang_of_Bandits
import numpy as np

# from util.io_helpers import log, out
# do not let classes log themselves.


class PolicyGradientAlgorithm:
    """
    REINFORCE / policy gradient for K-armed bandits using a softmax policy:

        pi_theta(a) = exp(theta[a]) / sum_b exp(theta[b])

    Update (no baseline):
        theta <- theta + alpha * reward * grad_theta log pi_theta(a)

    For softmax:
        grad log pi(a) = 1{a} - pi
    so componentwise:
        theta[i] <- theta[i] + alpha * reward * (1{i==a} - pi[i])
    """

    def __init__(self, bandit: Bandit, alpha: float):
        """
        Initialize policy gradient.

        Parameters
        ----------
        bandit : Bandit
            The bandit environment.
        alpha : float
            Learning rate / step size.
        """
        self.bandit = bandit
        self.alpha = float(alpha)

        # Number of arms
        self.n_arms = bandit.n_arms

        # Policy parameters theta
        self.theta = np.zeros(self.n_arms, dtype=float)

        # Tracking variables
        self.total_steps = 0
        self.arm_pull_counts = [0 for _ in range(self.n_arms)]
        self.arm_reward_sums = [0.0 for _ in range(self.n_arms)]


    def _policy_probs(self):
        # Numerically-stable softmax over theta
        x = self.theta - np.max(self.theta)
        ex = np.exp(x)
        s = float(np.sum(ex))
        if s <= 0.0 or not np.isfinite(s):
            return np.ones(self.n_arms, dtype=float) / float(self.n_arms)
        return ex / s

    def _select_arm(self) -> int:
        probs = self._policy_probs()
        return int(np.random.choice(self.n_arms, p=probs))

    def _update_theta(self, chosen_arm: int, reward: float):
        probs = self._policy_probs()

        # grad log pi(a) = e_a - pi
        grad = -probs
        grad[chosen_arm] += 1.0

        self.theta = self.theta + self.alpha * reward * grad

    def step(self):
        """
        Perform ONE step of policy gradient (no baseline).

        Returns
        -------
        chosen_arm : int
            Index of the arm pulled.
        reward : float
            Reward obtained from the bandit.
        """
        chosen_arm = self._select_arm()

        reward = self.bandit.pull(chosen_arm)

        # Update stats
        self.arm_pull_counts[chosen_arm] += 1
        self.arm_reward_sums[chosen_arm] += reward

        # Policy gradient update
        self._update_theta(chosen_arm, reward)

        self.total_steps += 1

        return chosen_arm, reward


class PolicyGradientBaselineAlgorithm:
    """
    REINFORCE with a baseline (variance reduction), still unbiased:

        theta <- theta + alpha * (reward - b_t) * grad log pi_theta(a)

    We use a simple running average baseline of rewards:
        b_t = (1/t) * sum_{s<=t} reward_s

    For softmax:
        grad log pi(a) = e_a - pi
    """

    def __init__(self, bandit: Bandit, alpha: float):
        """
        Initialize policy gradient with baseline.

        Parameters
        ----------
        bandit : Bandit
            The bandit environment.
        alpha : float
            Learning rate / step size.
        """
        self.bandit = bandit
        self.alpha = float(alpha)

        self.n_arms = bandit.n_arms

        # Policy parameters theta
        self.theta = np.zeros(self.n_arms, dtype=float)

        # Tracking variables
        self.total_steps = 0
        self.arm_pull_counts = [0 for _ in range(self.n_arms)]
        self.arm_reward_sums = [0.0 for _ in range(self.n_arms)]

        # Baseline tracking
        self.baseline = 0.0
        self.total_reward_sum = 0.0


    def _policy_probs(self):
        x = self.theta - np.max(self.theta)
        ex = np.exp(x)
        s = float(np.sum(ex))
        if s <= 0.0 or not np.isfinite(s):
            return np.ones(self.n_arms, dtype=float) / float(self.n_arms)
        return ex / s

    def _select_arm(self) -> int:
        probs = self._policy_probs()
        return int(np.random.choice(self.n_arms, p=probs))

    def _update_baseline(self, reward: float):
        # Running average baseline over all rewards observed so far
        self.total_reward_sum += reward
        t = self.total_steps + 1  # this step's index in 1..∞
        self.baseline = self.total_reward_sum / float(t)

    def _update_theta(self, chosen_arm: int, reward: float):
        probs = self._policy_probs()

        grad = -probs
        grad[chosen_arm] += 1.0

        advantage = reward - self.baseline
        self.theta = self.theta + self.alpha * advantage * grad

    def step(self):
        """
        Perform ONE step of policy gradient (with baseline).

        Returns
        -------
        chosen_arm : int
            Index of the arm pulled.
        reward : float
            Reward obtained from the bandit.
        """
        chosen_arm = self._select_arm()

        reward = self.bandit.pull(chosen_arm)

        # Update stats
        self.arm_pull_counts[chosen_arm] += 1
        self.arm_reward_sums[chosen_arm] += reward

        # Update baseline then policy
        self._update_baseline(reward)
        self._update_theta(chosen_arm, reward)

        self.total_steps += 1

        return chosen_arm, reward
    

class PolicyGradientBulkAlgorithm:
    """
    Bulk REINFORCE / policy gradient for K-armed bandits using a softmax policy:

        pi_theta(a) = exp(theta[a]) / sum_b exp(theta[b])

    Update (no baseline):
        theta <- theta + alpha * reward * grad_theta log pi_theta(a)

    For softmax:
        grad log pi(a) = e_a - pi
    """

    def __init__(self, bandit: Gang_of_Bandits, alpha: float):
        self.bandit = bandit
        self.alpha = float(alpha)

        self.n_bandits = bandit.n_bandits
        self.n_arms = bandit.n_arms

        # Policy parameters per bandit
        self.theta = np.zeros((self.n_bandits, self.n_arms), dtype=np.float64)

        # Tracking
        self.total_steps = 0
        self.arm_pull_counts = np.zeros((self.n_bandits, self.n_arms), dtype=np.int32)
        self.arm_reward_sums = np.zeros((self.n_bandits, self.n_arms), dtype=np.float64)

    def _policy_probs(self) -> np.ndarray:
        # Numerically-stable softmax per bandit (row-wise)
        x = self.theta - np.max(self.theta, axis=1, keepdims=True)
        ex = np.exp(x)
        s = np.sum(ex, axis=1, keepdims=True)

        # Fallback to uniform if degenerate (rare, but mirrors naive safety)
        bad = (~np.isfinite(s)) | (s <= 0.0)
        probs = ex / s
        if np.any(bad):
            probs[bad[:, 0], :] = 1.0 / float(self.n_arms)

        return probs

    def _select_arms(self) -> np.ndarray:
        # Sample from softmax using the Gumbel-max trick:
        # argmax_a (theta[a] + G_a) ~ softmax(theta)
        g = np.random.gumbel(loc=0.0, scale=1.0, size=(self.n_bandits, self.n_arms))
        return np.argmax(self.theta + g, axis=1).astype(np.int32)

    def _update_theta(self, chosen_arms: np.ndarray, rewards: np.ndarray) -> None:
        probs = self._policy_probs()  # (B, K)
        grad = -probs                 # (B, K)

        rows = np.arange(self.n_bandits)
        grad[rows, chosen_arms] += 1.0

        # theta <- theta + alpha * reward * grad
        self.theta = self.theta + (self.alpha * rewards)[:, None] * grad

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

        # Update stats (vectorized gather/scatter)
        rows = np.arange(self.n_bandits)
        self.arm_pull_counts[rows, chosen_arms] += 1
        self.arm_reward_sums[rows, chosen_arms] += rewards

        # Policy gradient update
        self._update_theta(chosen_arms, rewards)

        self.total_steps += 1
        return chosen_arms, rewards


class PolicyGradientBaselineBulkAlgorithm:
    """
    Bulk REINFORCE with a baseline (variance reduction), still unbiased:

        theta <- theta + alpha * (reward - b_t) * grad log pi_theta(a)

    Baseline is a running average reward PER bandit:
        b_t[b] = (1/t) * sum_{s<=t} reward_s[b]
    """

    def __init__(self, bandit: Gang_of_Bandits, alpha: float):
        self.bandit = bandit
        self.alpha = float(alpha)

        self.n_bandits = bandit.n_bandits
        self.n_arms = bandit.n_arms

        # Policy parameters per bandit
        self.theta = np.zeros((self.n_bandits, self.n_arms), dtype=np.float64)

        # Tracking
        self.total_steps = 0
        self.arm_pull_counts = np.zeros((self.n_bandits, self.n_arms), dtype=np.int32)
        self.arm_reward_sums = np.zeros((self.n_bandits, self.n_arms), dtype=np.float64)

        # Baseline tracking per bandit
        self.baseline = np.zeros(self.n_bandits, dtype=np.float64)
        self.total_reward_sum = np.zeros(self.n_bandits, dtype=np.float64)

    def _policy_probs(self) -> np.ndarray:
        x = self.theta - np.max(self.theta, axis=1, keepdims=True)
        ex = np.exp(x)
        s = np.sum(ex, axis=1, keepdims=True)

        bad = (~np.isfinite(s)) | (s <= 0.0)
        probs = ex / s
        if np.any(bad):
            probs[bad[:, 0], :] = 1.0 / float(self.n_arms)

        return probs

    def _select_arms(self) -> np.ndarray:
        # Gumbel-max sampling from softmax policy
        g = np.random.gumbel(loc=0.0, scale=1.0, size=(self.n_bandits, self.n_arms))
        return np.argmax(self.theta + g, axis=1).astype(np.int32)

    def _update_baseline(self, rewards: np.ndarray) -> None:
        # Running average baseline per bandit
        self.total_reward_sum += rewards
        t = self.total_steps + 1  # this step's index in 1..∞ (shared across bandits)
        self.baseline = self.total_reward_sum / float(t)

    def _update_theta(self, chosen_arms: np.ndarray, rewards: np.ndarray) -> None:
        probs = self._policy_probs()
        grad = -probs

        rows = np.arange(self.n_bandits)
        grad[rows, chosen_arms] += 1.0

        advantage = rewards - self.baseline
        self.theta = self.theta + (self.alpha * advantage)[:, None] * grad

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

        rows = np.arange(self.n_bandits)
        self.arm_pull_counts[rows, chosen_arms] += 1
        self.arm_reward_sums[rows, chosen_arms] += rewards

        # Baseline then policy update (matches naive order)
        self._update_baseline(rewards)
        self._update_theta(chosen_arms, rewards)

        self.total_steps += 1
        return chosen_arms, rewards
# src/etc.py
# This file contains
# - ETCAlgorithm (LEGACY, pull one arm at a time)
# - ETCBulkAlgorithm (CORE, supports bulk pulling)

from src.bandits import Bandit, Gang_of_Bandits
import numpy as np

# TESTING
# from util.io_helpers import log, out


class ETCBulkAlgorithm:
    """
    Bulk Explore-Then-Commit (ETC) algorithm.

    Assumption: all bandits use the same exploration schedule and therefore
    finish exploration at the same time (after exploration_rounds * n_arms steps).
    """

    def __init__(self, gang: Gang_of_Bandits, exploration_rounds: int):
        self.gang = gang
        self.exploration_rounds = int(exploration_rounds)

        self.n_bandits = gang.n_bandits
        self.n_arms = gang.n_arms

        if self.exploration_rounds <= 0:
            raise ValueError("explorati on_rounds must be a positive integer")

        # Total exploration steps
        self.exploration_steps_total = self.exploration_rounds * self.n_arms

        # Tracking
        self.total_steps = 0
        self.arm_pull_counts = np.zeros((self.n_bandits, self.n_arms), dtype=np.int32)
        self.arm_reward_sums = np.zeros((self.n_bandits, self.n_arms), dtype=np.float64)

        # Commitment
        self.committed_arm = np.full(self.n_bandits, -1, dtype=np.int32)
        self.committed = False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _in_exploration_phase(self) -> bool:
        return self.total_steps < self.exploration_steps_total

    def _select_exploration_arms(self) -> np.ndarray:
        # Round-robin: a_t = t mod K (same arm for every bandit on step t)
        arm = self.total_steps % self.n_arms
        return np.full(self.n_bandits, arm, dtype=np.int32)

    def _commit_to_best_arms(self) -> None:
        # Empirical means are well-defined because each arm has exactly exploration_rounds pulls.
        empirical_means = self.arm_reward_sums / self.exploration_rounds
        self.committed_arm = np.argmax(empirical_means, axis=1).astype(np.int32, copy=False)
        self.committed = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self):
        """
        Perform ONE bulk step.

        Returns
        -------
        chosen_arms : np.ndarray shape (n_bandits,)
        rewards : np.ndarray shape (n_bandits,)
        """
        if self._in_exploration_phase():
            chosen_arms = self._select_exploration_arms()
        else:
            if not self.committed:
                self._commit_to_best_arms()
            chosen_arms = self.committed_arm

        rewards = self.gang.bulk_pull(chosen_arms)

        rows = np.arange(self.n_bandits)
        self.arm_pull_counts[rows, chosen_arms] += 1
        self.arm_reward_sums[rows, chosen_arms] += rewards

        self.total_steps += 1
        return chosen_arms, rewards
    


# IMPORTANT: This is a working version and not part of the final assigment, some features might be legacy.
class ETCAlgorithm:

    def __init__(self, bandit: Bandit, exploration_rounds: int):
        """
        Initialize the ETC algorithm.

        Parameters
        ----------
        bandit : Bandit
            The bandit environment.
        exploration_rounds : int
            Number of exploration pulls per arm.
        """
        self.bandit = bandit
        self.exploration_rounds = exploration_rounds

        # Number of arms
        self.n_arms = bandit.n_arms

        # Tracking variables
        self.total_steps = 0
        self.arm_pull_counts = [0 for _ in range(self.n_arms)]
        self.arm_reward_sums = [0.0 for _ in range(self.n_arms)]

        # Commitment phase variables
        self.committed_arm = None


    def _in_exploration_phase(self) -> bool:
        # Check whether the algorithm is still in the exploration phase.

        for count in self.arm_pull_counts:
            if count < self.exploration_rounds:
                return True
        return False

    def _select_exploration_arm(self) -> int:
        # Select the next arm to explore using a round-robin rule: a_t = t mod K

        # total_steps starts at 0, which aligns naturally with 0-based arm indexing
        arm_index = self.total_steps % self.n_arms

        # Safety check: ensure we are still within exploration limits
        if self.arm_pull_counts[arm_index] < self.exploration_rounds:
            return arm_index

        raise RuntimeError("Exploration phase exceeded or no exploration arm available")

    def _commit_to_best_arm(self):
        # Determine and store the best arm based on empirical means.
        
        empirical_means = []

        for arm_index in range(self.n_arms):
            pulls = self.arm_pull_counts[arm_index]
            reward_sum = self.arm_reward_sums[arm_index]
            empirical_means.append(reward_sum / pulls)

        self.committed_arm = empirical_means.index(max(empirical_means))

        # log(f"Committed to arm {self.committed_arm +1} with empirical mean {max(empirical_means)} (out of {empirical_means})","etc.log")

    def step(self):
        """
        Perform ONE step of the ETC algorithm.

        Returns
        -------
        chosen_arm : int
            Index of the arm pulled.
        reward : float
            Reward obtained from the bandit.
        """
        # Decide which arm to pull
        if self._in_exploration_phase():
            chosen_arm = self._select_exploration_arm()
        else:
            if self.committed_arm is None:
                self._commit_to_best_arm()
            chosen_arm = self.committed_arm

        # Pull the arm
        reward = self.bandit.pull(chosen_arm)

        # Update tracking statistics
        self.arm_pull_counts[chosen_arm] += 1
        self.arm_reward_sums[chosen_arm] += reward
        self.total_steps += 1

        # log(f"Step {self.total_steps}: Pulled arm {chosen_arm}, reward={reward}","etc.log")

        return chosen_arm, reward




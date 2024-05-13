"""
Implements the Tabular MDP environment(s) from Duan et al., 2016
- 'RL^2 : Fast Reinforcement Learning via Slow Reinforcement Learning'.
"""

from typing import Tuple

from rl2.envs.abstract import MetaEpisodicEnv
from rl2.envs.stackelberg.matrix_game import IteratedMatrixGame

class FollowerEnv(MetaEpisodicEnv):
    
    def __init__(self, env: IteratedMatrixGame):

        self._env = env
        
        self.new_env()

        self._state = 0

    @property
    def max_episode_len(self):
        return self._env.episode_length

    @property
    def num_actions(self):
        """Get self._num_actions."""
        return self._env.action_space("follower").n

    @property
    def num_states(self):
        """Get self._num_states."""
        return self._env.observation_space("follower").n

    def _new_leader_policy(self):
        self._leader_response = [
            self._env.action_space("leader").sample() \
            for _ in range(self._env.observation_space("follower").n)
        ]

    def new_env(self) -> None:
        """
        Sample a new MDP from the distribution over MDPs.

        Returns:
            None
        """
        self._new_leader_policy()
        self._state = 0

    def reset(self) -> int:
        """
        Reset the environment.

        Returns:
            initial state.
        """
        self._state = self._env.reset()["follower"]
        return self._state

    def step(self, action, auto_reset=True) -> Tuple[int, float, bool, dict]:
        """
        Take action in the MDP, and observe next state, reward, done, etc.

        Args:
            action: action corresponding to an arm index.
            auto_reset: auto reset. if true, new_state will be from self.reset()

        Returns:
            new_state, reward, done, info.
        """

        a_ts = {"leader": self._leader_response[self._state],
               "follower": action}

        s_tp1s, r_ts, done_ts, _, _  = self._env.step(a_ts)
        s_tp1 = s_tp1s["follower"]
        self._state = s_tp1

        r_t = r_ts["follower"]

        done_t = done_ts["follower"]
        if done_t and auto_reset:
            s_tp1 = self.reset()

        return s_tp1, r_t, done_t, {}

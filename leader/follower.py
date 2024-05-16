import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pettingzoo.utils.wrappers import BaseParallelWrapper
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from envs.matrix_game import IteratedMatrixGame


# Wrapper that appends the leader's deterministic policy to the follower's observation
# Only works for small leader observation spaces
class FollowerWrapper(BaseParallelWrapper):
    def __init__(self, env, num_queries: int, leader_response: list | None = None):
        assert num_queries > 0, "num_queries must be greater than 0"
        assert leader_response is None or num_queries == len(
            leader_response
        ), "num_queries must be equal to the length of leader_response"

        super().__init__(env)
        self.num_queries = num_queries
        self.leader_response = leader_response

    def set_leader_response(self, leader_response: list):
        assert (
            len(leader_response) == self.num_queries
        ), "leader_response must be equal to the number of queries"
        self.leader_response = leader_response

    def observation_space(self, agent: str) -> spaces.Space:
        if agent == "leader":
            return self.env.observation_space(agent)

        leader_context_dims = [
            self.env.action_space("leader").n for _ in range(self.num_queries)
        ]

        if isinstance(self.env.observation_space(agent), spaces.Discrete):
            original_dims = [self.env.observation_space(agent).n]
        elif isinstance(self.env.observation_space(agent), spaces.MultiDiscrete):
            original_dims = self.env.observation_space(agent).nvec
        elif isinstance(self.env.observation_space(agent), spaces.MultiBinary):
            original_dims = [2] * self.env.observation_space(agent).n

        return spaces.MultiDiscrete([*original_dims, *leader_context_dims])

    def reset(self):
        obs = self.env.reset()
        if isinstance(obs["follower"], np.ndarray):
            obs["follower"] = np.concatenate(
                (obs["follower"], np.array(self.leader_response))
            )
        else:
            obs["follower"] = np.array([obs["follower"], *self.leader_response])
        return obs

    def step(self, actions):
        obs, rewards, terminated, truncated, infos = self.env.step(actions)
        if isinstance(obs["follower"], np.ndarray):
            obs["follower"] = np.concatenate(
                (obs["follower"], np.array(self.leader_response))
            )
        else:
            obs["follower"] = np.array([obs["follower"], *self.leader_response])
        return obs, rewards, terminated, truncated, infos


# Wrapper that adds the last action and reward to the follower's observation and plays multiple episodes
# Optionally zeros out the leaders reward in all episodes except the last one
# This allows Meta-RL using a recurrent policy
class FollowerWrapperMetaRL(BaseParallelWrapper):
    def __init__(
        self,
        env,
        num_episodes: int,
        max_reward: float = np.inf,
        min_reward: float = -np.inf,
        zero_leader_reward: bool = True,
        zero_follower_reward: bool = False,
    ):
        assert num_episodes >= 2, "num_episodes must be greater than or equal to 2"

        super().__init__(env)
        self.env = env
        self.num_episodes = num_episodes
        self.min_reward = min_reward
        self.max_reward = max_reward
        self.zero_leader_reward = zero_leader_reward
        self.zero_follower_reward = zero_follower_reward

        self.current_episode = 0
        self.current_step_in_episode = 0
        self.last_state = None
        self.reset_obs = False

    def observation_space(self, agent: str) -> spaces.Space:
        if agent == "leader":
            return self.env.observation_space(agent)

        # [last_leader_state, last_leader_action, episode, step_in_episode, original_observation]
        # -1 is used as a placeholder for the first step
        # TODO: alternatively, we could use a flag for the first step as in RL^2
        return gym.spaces.Box(
            low=np.array([-1, -1, 0, 0, 0]),
            high=np.array([4, 1, self.num_episodes - 1, np.inf, 4]),
        )

    # Start a fresh episode inside this trial
    # A trial is a sequence of episodes (as in RL^2)
    def _inner_reset(self):
        self.current_episode += 1
        self.current_step_in_episode = 0
        self.reset_next = False
        obs = self.env.reset()
        self.last_state = obs["leader"]
        # overwrite the follower's observation
        obs["follower"] = np.array(
            [
                -1,
                -1,
                self.current_episode - 1,
                self.current_step_in_episode,
                obs["follower"],
            ],
            dtype=np.float32,
        )

        return obs

    def reset(self):
        self.current_episode = 0
        self.current_step_in_episode = 0
        self.last_state = None
        self.reset_obs = False
        return self._inner_reset()

    def step(self, actions):
        # Reset next if true, if the last step returned terminated == True
        # and the current episode is not the last one
        if self.reset_next:
            obs = self._inner_reset()
            reward = {"follower": 0, "leader": 0}
            term = {"follower": False, "leader": False}
            trunc = {"follower": False, "leader": False}
            info = {"follower": {}, "leader": {}}
        else:
            self.current_step_in_episode += 1
            obs, reward, term, trunc, info = self.env.step(actions)
            last_leader_state = self.last_state
            self.last_state = obs["leader"]
            # overwrite the follower's observation
            obs["follower"] = np.array(
                [
                    last_leader_state,
                    actions["leader"],
                    self.current_episode - 1,
                    self.current_step_in_episode,
                    obs["follower"],
                ],
                dtype=np.float32,
            )

        if self.zero_leader_reward and self.current_episode < self.num_episodes:
            reward["leader"] = 0

        if self.zero_follower_reward and self.current_episode < self.num_episodes:
            reward["follower"] = 0

        # Only set terminated to True if the last episode is terminated
        terminated = False
        if any(term.values()) or any(trunc.values()):
            self.reset_next = True
            if self.current_episode == self.num_episodes:
                terminated = True
        term = {"follower": terminated, "leader": terminated}
        trunc = {"follower": False, "leader": False}

        return obs, reward, term, trunc, info

    def render():
        pass

    def close():
        pass


if __name__ == "__main__":
    env = IteratedMatrixGame(matrix="prisoners_dilemma", episode_length=10, memory=2)
    env = FollowerWrapperMetaRL(env, num_episodes=2)
    obs = env.reset()

    i = 1

    while True:
        i += 1
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        new_obs, rewards, terminated, truncated, infos = env.step(actions)

        print("STEP")
        print(obs)
        print(actions)
        print(rewards)
        print(new_obs)

        if any(terminated.values()) or any(truncated.values()):
            break

        obs = new_obs
    print("Done", i)

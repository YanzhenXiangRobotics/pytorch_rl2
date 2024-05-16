import gymnasium as gym
import numpy as np
import torch as tc

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class SingleAgentLeaderWrapper(gym.Env):
    def __init__(
        self,
        env,
        queries,
        follower_model,
        last_episode_in_trial,
        follower_epsilon_greedy: bool = False,
        epsilon: float = 0.1,
    ):
        self.env = env
        self.queries = queries
        self.follower_model = follower_model
        self.last_episode_in_trial = last_episode_in_trial
        self.epsilon = epsilon

        self.action_space = env.action_space("leader")
        self.observation_space = env.observation_space("leader")

        self.prev_follower_obs = None

    def reset(self, seed=None, options=None):
        self.current_step = 0

        self.prev_state = self.follower_model.initial_state(batch_size=1)
        obs = self.env.reset()
        self.prev_follower_obs = obs["follower"]
        self.prev_leader_obs = obs["leader"]

        return obs["leader"], {}

    def step(self, action):
        self.current_step += 1
        self.prev_leader_action = action

        if self.current_step < len(self.queries):
            return self.queries[self.current_step], 0, False, False, {}

        elif self.current_step == len(self.queries):
            obs = self.env.reset()
            return obs["leader"], 0, False, False, {}

        pi_dist_t, self.prev_state = self.follower_model(
            prev_leader_obs=tc.LongTensor(np.array([self.prev_leader_obs])),
            prev_leader_action=tc.LongTensor(np.array([self.prev_leader_action])),
            episode=tc.LongTensor(np.array([self.last_episode_in_trial])),
            step_in_episode=tc.LongTensor(np.array([self.current_step - len(self.queries) - 1])),
            curr_obs=tc.LongTensor(np.array([self.prev_follower_obs])),
            prev_state=self.prev_state,
        )
        follower_action = tc.argmax(pi_dist_t.probs).item()
        actions = {
            "leader": action,
            "follower": follower_action,
        }

        obs, reward, term, trunc, info = self.env.step(actions)
        
        self.prev_follower_obs = obs["follower"]
        self.prev_leader_obs = obs["leader"]

        return (
            obs["leader"],
            reward["leader"],
            term["leader"],
            trunc["leader"],
            info["leader"],
        )

    def render():
        pass

    def close():
        pass


# Wraps a multi-agent environment to a single-agent environment from the leader's perspective
# DOEST NOT prepend the initial segment to the leader's trajectory
# The queries are assumed to be fixed
class LeaderWrapperNoInitialSegment(gym.Env):
    def __init__(
        self,
        env,
        queries,
        follower_model,
        leader_model=None,
        follower_epsilon_greedy: bool = False,
        epsilon: float = 0.1,
        random_follower_policy_prob: float = 0.0,
    ):
        self.env = env
        self.queries = queries
        self.follower_model = follower_model
        self.leader_model = leader_model
        self.follower_epsilon_greedy = follower_epsilon_greedy
        self.epsilon = epsilon
        self.random_follower_policy_prob = random_follower_policy_prob

        self.last_follower_obs = None
        self.follower_policy = None

        self.action_space = env.action_space("leader")
        self.observation_space = env.observation_space("leader")

    def set_leader_model(self, leader_model):
        self.leader_model = leader_model

    def _get_next_follower_action(self):
        if self.follower_policy is not None:
            return self.follower_policy[self.last_follower_obs[0]]

        if self.follower_epsilon_greedy and np.random.rand() < self.epsilon:
            return self.env.action_space("follower").sample()
        else:
            follower_action, _states = self.follower_model.predict(
                self.last_follower_obs, deterministic=True
            )
            return follower_action

    def reset(self, seed=None, options=None):
        if (
            self.random_follower_policy_prob > 0
            and np.random.rand() < self.random_follower_policy_prob
        ):
            self.follower_policy = [
                self.env.action_space("follower").sample() for _ in range(5)
            ]
        else:
            self.follower_policy = None

        leader_response = [
            self.leader_model.predict(query)[0] for query in self.queries
        ]
        self.env.set_leader_response(leader_response)
        obs = self.env.reset()
        self.last_follower_obs = obs["follower"]
        return obs["leader"], {}

    def step(self, action):
        follower_action = self._get_next_follower_action()
        actions = {
            "leader": action,
            "follower": follower_action,
        }
        obs, reward, term, trunc, info = self.env.step(actions)
        self.last_follower_obs = obs["follower"]

        return (
            obs["leader"],
            reward["leader"],
            term["leader"],
            trunc["leader"],
            info["leader"],
        )

    def render():
        pass

    def close():
        pass

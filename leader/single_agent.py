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
        meta_episode_len,
        episode_len,
        follower_epsilon_greedy: bool = False,
        epsilon: float = 0.1,
    ):
        self.env = env
        self.queries = queries
        self.follower_model = follower_model
        self.meta_episode_len = meta_episode_len
        self.episode_len = episode_len
        self.epsilon = epsilon

        self.action_space = env.action_space("leader")
        self.observation_space = env.observation_space("leader")

        self.prev_follower_obs = None

        self.current_step = 0

    def reset(self, seed=None, options=None):

        self.prev_leader_obs = 0
        self.prev_leader_action = 0
        self.current_step = 0
        self.prev_state = self.follower_model.initial_state(batch_size=1)
        obs = self.env.reset()
        self.follower_obs = obs["follower"]

        return obs["leader"], {}
    
    def _inner_reset(self):
        curr_step = self.current_step
        obs_leader, _ = self.reset()
        self.current_step = curr_step
        return obs_leader, {}

    def step(self, action, auto_reset=True):

        self.current_step += 1

        pi_dist_t, self.prev_state = self.follower_model(
            prev_leader_obs=tc.LongTensor(np.array([self.prev_leader_obs])),
            prev_leader_action=tc.LongTensor(np.array([self.prev_leader_action])),
            episode=tc.LongTensor(np.array([int((self.current_step - 1) / self.episode_len)])),
            step_in_episode=tc.LongTensor(np.array([(self.current_step - 1) % self.episode_len])),
            curr_obs=tc.LongTensor(np.array([self.follower_obs])
                                   ),
            prev_state=self.prev_state.clone().detach(),
        )
        follower_action = tc.argmax(pi_dist_t.probs).item()
        actions = {
            "leader": action,
            "follower": follower_action,
        }   

        obs, reward, term, trunc, info = self.env.step(actions)

        if self.current_step <= (self.meta_episode_len - self.episode_len):
            reward["leader"] = 0

        if auto_reset and term["leader"]:
            obs["leader"], _ = self._inner_reset()

        # print(self.current_step - 1, self.prev_leader_obs, action, follower_action, obs["leader"], reward["leader"])
        
        self.prev_leader_obs = self.follower_obs
        self.prev_leader_action = action
        self.follower_obs = obs["follower"]

        return (
            obs["leader"],
            reward["leader"],
            self.current_step >= self.meta_episode_len,
            trunc["leader"],
            info["leader"],
        )

    def render():
        pass

    def close():
        pass
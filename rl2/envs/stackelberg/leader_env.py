import torch as tc
import numpy as np
import gymnasium as gym

from rl2.utils.constants import DEVICE


class SingleAgentLeaderWrapper(gym.Env):
    def __init__(
        self,
        env,
        follower_policy_net,
    ):
        self.env = env
        self.follower_policy_net = follower_policy_net

        self.action_space = env.action_space("leader")
        self.observation_space = env.observation_space("leader")

        self.follower_policy_net_hidden = None
        self.last_follower_obs = None
        self.last_follower_action = None
        self.last_follower_reward = None
        self.last_follower_done = None

    def _get_next_follower_action(self):
        pi_dist, hidden = self.follower_policy_net(
            curr_obs=tc.LongTensor(self.last_follower_obs).to(DEVICE),
            prev_action=tc.LongTensor(self.last_follower_action).to(DEVICE),
            prev_reward=tc.FloatTensor(self.last_follower_reward).to(DEVICE),
            prev_done=tc.FloatTensor(self.last_follower_done).to(DEVICE),
            prev_state=self.follower_policy_net_hidden,
        )
        self.follower_policy_net_hidden = hidden
        action = tc.atleast_1d(tc.argmax(pi_dist.probs))
        action = action.squeeze(0).detach().cpu().numpy()
        return action

    def reset(self, seed=None, options=None):
        self.follower_policy_net_hidden = self.follower_policy_net.initial_state(
            batch_size=1
        )
        self.last_follower_action = np.array([0])
        self.last_follower_reward = np.array([0.0])
        self.last_follower_done = np.array([1.0])
        obs = self.env.reset()
        self.last_follower_obs = np.array([obs["follower"]])
        return obs["leader"], {}

    def step(self, action):
        follower_action = self._get_next_follower_action()
        actions = {
            "leader": action,
            "follower": follower_action,
        }
        obs, reward, term, trunc, info = self.env.step(actions)
        self.last_follower_obs = np.array([obs["follower"]])
        self.last_follower_action = np.array([follower_action])
        self.last_follower_reward = np.array([reward["follower"]])
        self.last_follower_done = np.array([float(term["follower"])])

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

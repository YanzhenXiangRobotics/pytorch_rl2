"""
Implements common algorithmic components for training
stateful meta-reinforcement learning agents.
"""

import torch as tc
import numpy as np

from rl2.envs.abstract import MetaEpisodicEnv
from rl2.agents.integration.policy_net import StatefulPolicyNet
from rl2.agents.integration.value_net import StatefulValueNet
from rl2.utils.constants import DEVICE


class MetaEpisode:
    def __init__(self):
        self.horizon = 0
        self.obs = np.array([])
        self.acs = np.array([], dtype='int64')
        self.rews = np.array([], dtype='float32')
        self.dones = np.array([], dtype='float32')
        self.logpacs = np.array([], dtype='float32')
        self.vpreds = np.array([], dtype='float32')
        self.advs = np.array([], dtype='float32')
        self.tdlam_rets = np.array([], dtype='float32')

    def add_step(self, obs, ac, rew, done, logpac, vpred):
        if self.horizon == 0:
            self.obs = np.array([obs])
        else:
            self.obs = np.append(self.obs, obs)
        self.acs = np.append(self.acs, ac)
        self.rews = np.append(self.rews, rew)
        self.dones = np.append(self.dones, done)
        self.logpacs = np.append(self.logpacs, logpac)
        self.vpreds = np.append(self.vpreds, vpred)

        self.horizon += 1

@tc.no_grad()
def generate_meta_episode(
        env: MetaEpisodicEnv,
        policy_net: StatefulPolicyNet,
        value_net: StatefulValueNet,
        num_episodes: int
    ) -> MetaEpisode:
    """
    Generates a meta-episode: a sequence of episodes concatenated together,
    with decisions being made by a recurrent agent with state preserved
    across episode boundaries.

    Args:
        env: environment.
        policy_net: policy network.
        value_net: value network.
        num_episodes: episodes per trial.

    Returns:
        meta_episode: an instance of the meta-episode class.
    """

    env.new_env()
    meta_episode = MetaEpisode()

    o_t = np.array([env.reset()])
    a_tm1 = np.array([0])
    r_tm1 = np.array([0.0])
    d_tm1 = np.array([1.0])
    h_tm1_policy_net = policy_net.initial_state(batch_size=1)
    h_tm1_value_net = value_net.initial_state(batch_size=1)

    current_episode = 0

    while True:
        pi_dist_t, h_t_policy_net = policy_net(
            curr_obs=o_t,
            prev_action=tc.LongTensor(a_tm1).to(DEVICE),
            prev_reward=tc.FloatTensor(r_tm1).to(DEVICE),
            prev_done=tc.FloatTensor(d_tm1).to(DEVICE),
            prev_state=h_tm1_policy_net)

        vpred_t, h_t_value_net = value_net(
            curr_obs=o_t,
            prev_action=tc.LongTensor(a_tm1).to(DEVICE),
            prev_reward=tc.FloatTensor(r_tm1).to(DEVICE),
            prev_done=tc.FloatTensor(d_tm1).to(DEVICE),
            prev_state=h_tm1_value_net)

        a_t = pi_dist_t.sample()
        log_prob_a_t = pi_dist_t.log_prob(a_t)

        o_tp1, r_t, done_t, _ = env.step(
            action=a_t.squeeze(0).detach().cpu().numpy(),
            auto_reset=True)

        meta_episode.add_step(
            obs=o_t[0],
            ac=a_t.squeeze(0).detach().cpu().numpy(),
            rew=r_t,
            done=float(done_t),
            logpac=log_prob_a_t.squeeze(0).detach().cpu().numpy(),
            vpred=vpred_t.squeeze(0).detach().cpu().numpy())

        o_t = np.array([o_tp1])
        a_tm1 = np.array([meta_episode.acs[-1]])
        r_tm1 = np.array([meta_episode.rews[-1]])
        d_tm1 = np.array([meta_episode.dones[-1]])
        h_tm1_policy_net = h_t_policy_net
        h_tm1_value_net = h_t_value_net

        if done_t:
            current_episode += 1
            if current_episode >= num_episodes:
                break

    return meta_episode


@tc.no_grad()
def assign_credit(
        meta_episode: MetaEpisode,
        gamma: float,
        lam: float
    ) -> MetaEpisode:
    """
    Compute td lambda returns and generalized advantage estimates.

    Note that in the meta-episodic setting of RL^2, the objective is
    to maximize the expected discounted return of the meta-episode,
    so we do not utilize the usual 'done' masking in this function.

    Args:
        meta_episode: meta-episode.
        gamma: discount factor.
        lam: GAE decay parameter.

    Returns:
        meta_episode: an instance of the meta-episode class,
        with generalized advantage estimates and td lambda returns computed.
    """

    meta_episode.advs = np.zeros_like(meta_episode.acs, dtype='float32')
    meta_episode.tdlam_rets = np.zeros_like(meta_episode.acs, dtype='float32')

    T = len(meta_episode.acs)
    for t in reversed(range(0, T)):  # T-1, ..., 0.
        r_t = meta_episode.rews[t]
        V_t = meta_episode.vpreds[t]
        V_tp1 = meta_episode.vpreds[t+1] if t+1 < T else 0.0
        A_tp1 = meta_episode.advs[t+1] if t+1 < T else 0.0
        delta_t = -V_t + r_t + gamma * V_tp1
        A_t = delta_t + gamma * lam * A_tp1
        meta_episode.advs[t] = A_t

    meta_episode.tdlam_rets = meta_episode.vpreds + meta_episode.advs
    return meta_episode


def huber_func(y_pred, y_true, delta=1.0):
    a = y_pred-y_true
    a_abs = tc.abs(a)
    a2 = tc.square(a)
    terms = tc.where(
        tc.less(a_abs, delta * tc.ones_like(a2).to(DEVICE)),
        0.5 * a2,
        delta * (a_abs - 0.5 * delta)
    )
    return terms

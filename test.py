"""
Script for training stateful meta-reinforcement learning agents
"""

import argparse
from functools import partial

import torch as tc
import numpy as np

from rl2.utils.constants import DEVICE
from rl2.utils.setup_experiment import create_env, get_policy_net_for_inference


def create_argparser():
    parser = argparse.ArgumentParser(
        description="""Training script for RL^2.""")

    ### Environment
    parser.add_argument("--environment", choices=['bandit', 'tabular_mdp', 'matrix_game'],
                        default='bandit')
    parser.add_argument("--num_states", type=int, default=10,
                        help="Ignored if environment is bandit.")
    parser.add_argument("--num_actions", type=int, default=5)
    parser.add_argument("--max_episode_len", type=int, default=10,
                        help="Timesteps before automatic episode reset. " +
                             "Ignored if environment is bandit.")
    parser.add_argument("--meta_episode_len", type=int, default=100,
                        help="Timesteps per meta-episode.")

    ### Architecture
    parser.add_argument(
        "--architecture", choices=['gru', 'lstm', 'snail', 'transformer'],
        default='gru')
    parser.add_argument("--num_features", type=int, default=256)

    ### Checkpointing
    parser.add_argument("--model_name", type=str, default='defaults')
    parser.add_argument("--checkpoint_dir", type=str, default='checkpoints')
    return parser

def main():
    args = create_argparser().parse_args()

    # create env.
    env = create_env(
        environment=args.environment,
        num_states=args.num_states,
        num_actions=args.num_actions,
        max_episode_len=args.max_episode_len)

    policy_net = get_policy_net_for_inference(args)

    env._leader_response = [1, 0, 0, 1, 1]
    action = np.array([0])
    reward = np.array([0.0])
    done = np.array([1.0])
    obs = np.array([env.reset()])
    hidden = policy_net.initial_state(batch_size=1)
    for _ in range(args.meta_episode_len):
        pi_dist, hidden = policy_net(
            curr_obs=tc.LongTensor(obs).to(DEVICE),
            prev_action=tc.LongTensor(action).to(DEVICE),
            prev_reward=tc.FloatTensor(reward).to(DEVICE),
            prev_done=tc.FloatTensor(done).to(DEVICE),
            prev_state=hidden,
        )
        action = tc.atleast_1d(tc.argmax(pi_dist.probs))

        new_obs, reward, done, _ = env.step(
            action=action.squeeze(0).detach().cpu().numpy(),
            auto_reset=True
        )

        print(obs[0], action.squeeze(0).detach().cpu().numpy(), reward)

        obs = np.array([new_obs])
        action = action.detach().cpu().numpy()
        reward = np.array([reward])
        done = np.array([float(done)])

if __name__ == '__main__':
    main()

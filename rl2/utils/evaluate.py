import numpy as np
import torch as tc

from rl2.utils.setup_experiment import get_policy_net_for_inference, create_env
from rl2.utils.constants import DEVICE

# Only works for matrix game at the moment
def evaluate(args, policy_net=None, leader_policy=None, verbose=False):
    # create env.
    env = create_env(
        name=args.environment,
        num_states=args.num_states,
        num_actions=args.num_actions,
        max_episode_len=args.max_episode_len,
        headless=args.headless)

    if policy_net is None:
        policy_net = get_policy_net_for_inference(args, env)

    def evaluate_policy(leader_policy):
        rewards = []

        env._leader_response = leader_policy

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

            rewards.append(reward)

            if verbose:
                print(obs[0], action.squeeze(0).detach().cpu().numpy(), reward)

            obs = np.array([new_obs])
            action = action.detach().cpu().numpy()
            reward = np.array([reward])
            done = np.array([float(done)])
        
        return np.sum(rewards)

    if leader_policy is not None:
        return evaluate_policy(leader_policy)
    else:
        rewards = []
        for i in range(32):
            reward = evaluate_policy([int(x) for x in np.binary_repr(i, width=5)])
            rewards.append(reward)
        return np.mean(rewards)


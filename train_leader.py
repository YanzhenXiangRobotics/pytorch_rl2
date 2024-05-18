import argparse

# import ppo form stable baselines
from stable_baselines3 import PPO

from rl2.utils.setup_experiment import get_policy_net_for_inference

from rl2.envs.stackelberg.matrix_game import IteratedMatrixGame
from rl2.envs.stackelberg.trial_wrapper import TrialWrapper
from rl2.envs.stackelberg.leader_env import SingleAgentLeaderWrapper


def create_argparser():
    parser = argparse.ArgumentParser(
        description="""Training script for RL^2.""")

    ### Environment
    parser.add_argument("--environment", choices=['matrix_game'],
                        default='matrix_game')
    parser.add_argument("--max_episode_len", type=int, default=10,
                        help="Timesteps before automatic episode reset. " +
                             "Ignored if environment is bandit.")
    parser.add_argument("--meta_episode_len", type=int, default=30,
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

    policy_net = get_policy_net_for_inference(args)

    # create env.
    env = IteratedMatrixGame(matrix='prisoners_dilemma',
                             episode_length=args.max_episode_len,
                             memory=2)
    env = TrialWrapper(env, num_episodes=3)
    env = SingleAgentLeaderWrapper(env, follower_policy_net=policy_net)

    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=1000)

    obs, _ = env.reset()
    while True:
        action = model.predict(obs, deterministic=True)[0]
        new_obs, reward, terminated, truncated, info = env.step(action)
        print(obs, action, reward)
        obs = new_obs
        if terminated or truncated:
            break

    leader_policy = [model.predict(obs, deterministic=True)[0].item() for obs in range(5)]
    print(leader_policy)

if __name__ == '__main__':
    main()

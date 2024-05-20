import argparse

import numpy as np

# import ppo form stable baselines
from stable_baselines3 import PPO

from rl2.utils.setup_experiment import create_env, get_policy_net_for_inference

from rl2.envs.stackelberg.matrix_game import IteratedMatrixGame
from rl2.envs.stackelberg.trial_wrapper import TrialWrapper
from rl2.envs.stackelberg.leader_env import SingleAgentLeaderWrapper

import wandb
from wandb.integration.sb3 import WandbCallback

def create_argparser():
    parser = argparse.ArgumentParser(description="""Training script for RL^2.""")

    ### Environment
    parser.add_argument(
        "--environment",
        choices=["matrix_game_follower", "drone_game_follower"],
        default="matrix_game_follower",
    )
    parser.add_argument(
        "--max_episode_len",
        type=int,
        default=10,
        help="Timesteps before automatic episode reset. "
        + "Ignored if environment is bandit.",
    )
    parser.add_argument(
        "--meta_episode_len", type=int, default=30, help="Timesteps per meta-episode."
    )

    ### Architecture
    parser.add_argument(
        "--architecture", choices=["gru", "lstm", "snail", "transformer"], default="gru"
    )
    parser.add_argument("--num_features", type=int, default=256)

    ### Checkpointing
    parser.add_argument("--model_name", type=str, default="defaults")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    return parser


def main():

    run = wandb.init(project="rl2-leader", sync_tensorboard=True)

    args = create_argparser().parse_args()

    follower_env = create_env(
        name=args.environment,
        max_episode_len=args.max_episode_len,
        headless=True,
    )

    policy_net = get_policy_net_for_inference(args, follower_env)

    env = TrialWrapper(follower_env._env, num_episodes=3)
    env = SingleAgentLeaderWrapper(env, follower_policy_net=policy_net)

    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=f"runs/{run.id}",)
    model.learn(total_timesteps=600_000, callback=WandbCallback(gradient_save_freq=100, verbose=2))

    model.save(f"checkpoints/leader_ppo_{args.environment}")

    if args.environment == "drone_game_follower":
        env.env.env.headless = False

    obs, _ = env.reset()
    while True:
        action = model.predict(obs, deterministic=True)[0]
        new_obs, reward, terminated, truncated, info = env.step(action)
        print(obs, action, reward)
        obs = new_obs
        if terminated or truncated:
            break
    if args.environment == "matrix_game_follower":
        leader_policy = [
            model.predict(obs, deterministic=True)[0].item() for obs in range(5)
        ]
        print(leader_policy)
    elif args.environment == "drone_game_follower":
        leader_policy = [
            model.predict(
                [int(b) for b in np.binary_repr(obs, width=4)], deterministic=True
            )[0].item()
            for obs in range(16)
        ]
        print(leader_policy)


if __name__ == "__main__":
    main()

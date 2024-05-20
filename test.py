"""
Script for training stateful meta-reinforcement learning agents
"""

import argparse
from rl2.utils.evaluate import evaluate


def create_argparser():
    parser = argparse.ArgumentParser(description="""Training script for RL^2.""")

    parser.add_argument(
        "--headless",
        action="store_true",
        help="Whether to display GUI. Only for drone game.",
    )

    ### Environment
    parser.add_argument(
        "--environment",
        choices=[
            "bandit",
            "tabular_mdp",
            "matrix_game_follower",
            "drone_game_follower",
        ],
        default="bandit",
    )
    parser.add_argument(
        "--num_states", type=int, default=10, help="Ignored if environment is bandit."
    )
    parser.add_argument("--num_actions", type=int, default=5)
    parser.add_argument(
        "--max_episode_len",
        type=int,
        default=10,
        help="Timesteps before automatic episode reset. "
        + "Ignored if environment is bandit.",
    )
    parser.add_argument(
        "--meta_episode_len", type=int, default=100, help="Timesteps per meta-episode."
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
    args = create_argparser().parse_args()
    if args.environment == "matrix_game_follower":
        evaluate(args, verbose=True, leader_policy=[1, 0, 0, 1, 1])
    elif args.environment == "drone_game_follower":
        evaluate(
            args,
            verbose=True,
            # leader_policy=[0, 3, 3, 0, 3, 0, 3, 0, 0, 3, 0, 3, 0, 3, 0, 3],
        leader_policy=[0 for _ in range(16)]
        )


if __name__ == "__main__":
    main()

"""
Script for training stateful meta-reinforcement learning agents
"""

import os
import yaml
from rl2.utils.evaluate import evaluate

from train import add_args


def main():
    file_dir = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(file_dir, "rl2", "envs", "config.yml"), "rb") as file:
        config = yaml.safe_load(file.read())
    config = add_args(config)
    if config.env.name == "matrix_game_follower":
        evaluate(config, verbose=True, leader_policy=[1, 0, 0, 1, 1])
    elif config.env.name == "drone_game_follower":
        evaluate(
            config,
            verbose=True,
            leader_policy=[10 for _ in range(2**12)]
        )


if __name__ == "__main__":
    main()

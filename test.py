"""
Script for training stateful meta-reinforcement learning agents
"""

import os
import yaml
import numpy as np


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
            # leader_policy=[1, 1, 1, 1] + [np.random.randint(2) for _ in range(2**10-4)],
            # leader_policy=[0 for _ in range(2**4)]
            leader_policy= [0, 3, 0, 3, 3, 0, 3, 0, 0, 0, 3, 3, 0, 3, 0, 3]
            # leader_policy= [1, 3, 1, 3, 3, 1, 3, 1, 1, 1, 3, 3, 1, 3, 1, 3]
        )


if __name__ == "__main__":
    main()

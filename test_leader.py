import os

import yaml

import numpy as np

# import ppo form stable baselines
from stable_baselines3 import PPO

from rl2.utils.setup_experiment import create_env, get_policy_net_for_inference
from rl2.utils.checkpoint_util import _latest_step

from rl2.envs.stackelberg.trial_wrapper import TrialWrapper
from rl2.envs.stackelberg.leader_env import SingleAgentLeaderWrapper

from train import add_args

def test(env, config):

    model_dir = "checkpoints/leader"
    latest_step = _latest_step(model_dir)
    model = PPO.load(f"checkpoints/leader/model_{latest_step}_steps.zip")

    if config.env.name == "drone_game_follower":
        env.env.env.headless = False
        env.env.env.sleep_time = 0.5

    obs, _ = env.reset()
    while True:
        action = model.predict(obs, deterministic=True)[0]
        new_obs, reward, terminated, truncated, _ = env.step(action)
        print(obs, action, reward)
        obs = new_obs
        if terminated or truncated:
            break

    if config.env.name == "matrix_game_follower":
        leader_policy = [
            model.predict(obs, deterministic=True)[0].item() for obs in range(5)
        ]
        print(leader_policy)
    elif config.env.name == "drone_game_follower":
        leader_policy = [
            model.predict(
                [int(b) for b in np.binary_repr(obs, width=4)], deterministic=True
            )[0].item()
            for obs in range(16)
        ]
        print(leader_policy)

if __name__ == "__main__":
    file_dir = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(file_dir, "rl2", "envs", "config.yml"), "rb") as file:
        config = yaml.safe_load(file.read())

    config = add_args(config)

    follower_env = create_env(config=config)

    policy_net = get_policy_net_for_inference(follower_env, config)

    env = TrialWrapper(follower_env._env, num_episodes=3)
    env = SingleAgentLeaderWrapper(env, follower_policy_net=policy_net)
    
    test(env, config)
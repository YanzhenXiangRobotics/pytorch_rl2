"""
Script for training stateful meta-reinforcement learning agents
"""

import argparse
from functools import partial

import torch as tc
import numpy as np

from rl2.envs.stackelberg.follower_env import FollowerEnv, IteratedMatrixGame

from rl2.agents.preprocessing.tabular import MABPreprocessing, MDPPreprocessing
from rl2.agents.architectures.gru import GRU
from rl2.agents.architectures.lstm import LSTM
from rl2.agents.architectures.snail import SNAIL
from rl2.agents.architectures.transformer import Transformer
from rl2.agents.heads.policy_heads import LinearPolicyHead
from rl2.agents.heads.value_heads import LinearValueHead
from rl2.agents.integration.policy_net import StatefulPolicyNet
from rl2.agents.integration.value_net import StatefulValueNet
from rl2.algos.ppo import training_loop

from rl2.utils.checkpoint_util import maybe_load_checkpoint, save_checkpoint
from rl2.utils.comm_util import get_comm, sync_state
from rl2.utils.constants import ROOT_RANK
from rl2.utils.optim_util import get_weight_decay_param_groups

from leader.single_agent import SingleAgentLeaderWrapper

from stable_baselines3 import PPO

import wandb
from wandb.integration.sb3 import WandbCallback

def create_argparser():
    parser = argparse.ArgumentParser(description="""Training script for RL^2.""")

    ### Environment
    parser.add_argument(
        "--environment",
        choices=["bandit", "tabular_mdp", "matrix_game"],
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

    ### Training
    parser.add_argument("--max_pol_iters", type=int, default=12000)
    parser.add_argument(
        "--meta_episodes_per_policy_update",
        type=int,
        default=-1,
        help="If -1, quantity is determined using a formula",
    )
    parser.add_argument("--meta_episodes_per_learner_batch", type=int, default=60)
    parser.add_argument("--ppo_opt_epochs", type=int, default=8)
    parser.add_argument("--ppo_clip_param", type=float, default=0.10)
    parser.add_argument("--ppo_ent_coef", type=float, default=0.01)
    parser.add_argument("--discount_gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.3)
    parser.add_argument("--standardize_advs", type=int, choices=[0, 1], default=0)
    parser.add_argument("--adam_lr", type=float, default=2e-4)
    parser.add_argument("--adam_eps", type=float, default=1e-5)
    parser.add_argument("--adam_wd", type=float, default=0.01)
    parser.add_argument("--resume", default=False, action="store_true")
    return parser


def create_env(environment, max_episode_len):
    if environment == "matrix_game":
        return FollowerEnv(
            env=IteratedMatrixGame(
                matrix="prisoners_dilemma", episode_length=max_episode_len, memory=2
            )
        )
    raise NotImplementedError


def create_preprocessing(
    environment,
    num_states,
    num_actions,
    num_episodes_per_trial,
    episode_len,
):
    if environment == "matrix_game":
        return MDPPreprocessing(
            num_states=num_states,
            num_actions=num_actions,
            num_episodes_per_trial=num_episodes_per_trial,
            episode_len=episode_len,
        )
    raise NotImplementedError


def create_architecture(architecture, input_dim, num_features, context_size):
    if architecture == "gru":
        return GRU(
            input_dim=input_dim,
            hidden_dim=num_features,
            forget_bias=1.0,
            use_ln=True,
            reset_after=True,
        )
    if architecture == "lstm":
        return LSTM(
            input_dim=input_dim, hidden_dim=num_features, forget_bias=1.0, use_ln=True
        )
    if architecture == "snail":
        return SNAIL(
            input_dim=input_dim,
            feature_dim=num_features,
            context_size=context_size,
            use_ln=True,
        )
    if architecture == "transformer":
        return Transformer(
            input_dim=input_dim,
            feature_dim=num_features,
            n_layer=9,
            n_head=2,
            n_context=context_size,
        )
    raise NotImplementedError


def create_head(head_type, num_features, num_actions):
    if head_type == "policy":
        return LinearPolicyHead(num_features=num_features, num_actions=num_actions)
    if head_type == "value":
        return LinearValueHead(num_features=num_features)
    raise NotImplementedError


def create_net(
    net_type,
    environment,
    architecture,
    num_states,
    num_actions,
    num_features,
    num_episodes_per_trial,
    episode_len,
    context_size,
):
    preprocessing = create_preprocessing(
        environment=environment, 
        num_states=num_states, 
        num_actions=num_actions,
        num_episodes_per_trial=num_episodes_per_trial,
        episode_len=episode_len,
    )
    architecture = create_architecture(
        architecture=architecture,
        input_dim=preprocessing.output_dim,
        num_features=num_features,
        context_size=context_size,
    )
    head = create_head(
        head_type=net_type,
        num_features=architecture.output_dim,
        num_actions=num_actions,
    )

    if net_type == "policy":
        return StatefulPolicyNet(
            preprocessing=preprocessing, architecture=architecture, policy_head=head
        )
    if net_type == "value":
        return StatefulValueNet(
            preprocessing=preprocessing, architecture=architecture, value_head=head
        )
    raise NotImplementedError


def train(args, leader_env):
    

    leader_run = wandb.init(project="rl2-matgame-leader", sync_tensorboard=True)
    leader_model = PPO(
        "MlpPolicy", 
        leader_env, 
        verbose=1, 
        tensorboard_log=f"runs/{leader_run.id}", 
        gamma=0.9999
        # learning_rate=1e-2,
        # ent_coef=1e-2,
    )
    if args.resume:
        print("Resuming...\n")
        leader_model = PPO.load("checkpoints/leader_ppo", env=leader_env)
    leader_model.learn(
        total_timesteps=80_000,
        callback=WandbCallback(gradient_save_freq=100, verbose=2),
    )
    leader_model.save("checkpoints/leader_ppo")

def test(leader_env):

    leader_model = PPO.load("checkpoints/leader_ppo", env=leader_env)
    # play a single episode to check learned leader and follower policies
    obs, _ = leader_env.reset()
    while True:
        action = leader_model.predict(obs, deterministic=True)[0]
        new_obs, rewards, terminated, truncated, _ = leader_env.step(action)
        print(leader_env.current_step, obs, action, rewards)
        obs = new_obs

        if terminated or truncated:
            break

if __name__ == "__main__":

    args = create_argparser().parse_args()
    
    follower_env = create_env(environment=args.environment, max_episode_len=args.max_episode_len)

    follower_policy_net = create_net(
        net_type="policy",
        environment=args.environment,
        architecture=args.architecture,
        num_states=follower_env.num_states,
        num_actions=follower_env.num_actions,
        num_episodes_per_trial=int(args.meta_episode_len / args.max_episode_len),
        episode_len=args.max_episode_len,
        num_features=args.num_features,
        context_size=args.meta_episode_len,
    )

    comm = get_comm()

    follower_policy_optimizer = tc.optim.AdamW(
        get_weight_decay_param_groups(follower_policy_net, args.adam_wd),
        lr=args.adam_lr,
        eps=args.adam_eps,
    )

    follower_policy_scheduler = None

    # load checkpoint, if applicable.
    pol_iters_so_far = 0
    if comm.Get_rank() == ROOT_RANK:
        a = maybe_load_checkpoint(
            checkpoint_dir=args.checkpoint_dir,
            model_name=f"{args.model_name}/policy_net",
            model=follower_policy_net,
            optimizer=follower_policy_optimizer,
            scheduler=follower_policy_scheduler,
            steps=None,
        )

        pol_iters_so_far = a

    # sync state.
    pol_iters_so_far = comm.bcast(pol_iters_so_far, root=ROOT_RANK)
    sync_state(
        model=follower_policy_net,
        optimizer=follower_policy_optimizer,
        scheduler=follower_policy_scheduler,
        comm=comm,
        root=ROOT_RANK,
    )

    leader_env = follower_env._env
    leader_env = SingleAgentLeaderWrapper(
        leader_env,
        queries=[0, 1, 2, 3, 4],
        follower_model=follower_policy_net,
        meta_episode_len=args.meta_episode_len,
        episode_len=args.max_episode_len,
    )

    # follower_env._leader_response = [1,0,0,1,1]
    # ol_tm1 = np.array([0])
    # al_tm1 = np.array([0])
    # obs = np.array([follower_env.reset()])
    # hidden = follower_policy_net.initial_state(batch_size=1)
    # for t in range(args.meta_episode_len):

    #     ep_t = np.array([int(t / args.max_episode_len)])
    #     st_t = np.array([t % args.max_episode_len])
    #     pi_dist, hidden = follower_policy_net(
    #         prev_leader_obs=tc.LongTensor(ol_tm1),
    #         prev_leader_action=tc.LongTensor(al_tm1),
    #         episode=tc.LongTensor(ep_t),
    #         step_in_episode=tc.LongTensor(st_t),
    #         curr_obs=tc.LongTensor(obs),
    #         prev_state=hidden
    #     )
    #     action = tc.atleast_1d(tc.argmax(pi_dist.probs))

    #     al_t, obs_next, reward, done, _ = follower_env.step(
    #         action=action.squeeze(0).detach().numpy(),
    #         auto_reset=True
    #     )

    #     if t < (args.meta_episode_len - args.max_episode_len):
    #         reward = 0

    #     print(t, obs[0], al_t, action.squeeze(0).item(), obs_next, reward)
        
    #     al_tm1 = np.array([al_t])
    #     ol_tm1 = obs
    #     obs = obs_next
    #     obs = np.array([obs])
    #     reward = np.array([reward])
    #     done = np.array([float(done)])

    # train(args, leader_env)

    test(leader_env)

"""
Script for training stateful meta-reinforcement learning agents
"""

import argparse
from functools import partial

import torch as tc

from rl2.agents.architectures.gru import GRU
from rl2.agents.architectures.lstm import LSTM
from rl2.agents.architectures.snail import SNAIL
from rl2.agents.architectures.transformer import Transformer
from rl2.agents.heads.policy_heads import LinearPolicyHead
from rl2.agents.heads.value_heads import LinearValueHead
from rl2.algos.ppo import training_loop

from rl2.utils.checkpoint_util import maybe_load_checkpoint, save_checkpoint
from rl2.utils.comm_util import get_comm, sync_state
from rl2.utils.constants import ROOT_RANK, DEVICE
from rl2.utils.optim_util import get_weight_decay_param_groups
from rl2.utils.setup_experiment import create_env, create_net


def create_argparser():
    parser = argparse.ArgumentParser(description="""Training script for RL^2.""")

    parser.add_argument(
        "--log_wandb", action="store_true", help="Whether to use wandb for logging."
    )

    parser.add_argument(
        "--headless", action="store_true", help="Whether to display GUI. Only for drone game."
    )

    ### Environment
    parser.add_argument(
        "--environment",
        choices=["bandit", "tabular_mdp", "matrix_game_follower", "drone_game_follower"],
        default="bandit",
    )
    parser.add_argument(
        "--num_states", type=int, default=10, help="Only for yabular mdp."
    )
    parser.add_argument(
        "--num_actions", type=int, default=5, help="Only for badit and tabular mdp."
    )
    parser.add_argument(
        "--max_episode_len",
        type=int,
        default=10,
        help="Timesteps before automatic episode reset. "
        + "Ignored if environment is bandit.",
    )
    parser.add_argument(
        "--num_meta_episodes", type=int, default=3, help="Episode per meta-episode."
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
    return parser

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

def main():
    print("Using device:", DEVICE)

    args = create_argparser().parse_args()
    comm = get_comm()

    # create env.
    env = create_env(
        name=args.environment,
        num_states=args.num_states,
        num_actions=args.num_actions,
        max_episode_len=args.max_episode_len,
        headless=args.headless
    )

    # create learning system.
    policy_net = create_net(
        net_type="policy",
        env=env,
        architecture=args.architecture,
        num_features=args.num_features,
        context_size=0,
    )

    value_net = create_net(
        net_type="value",
        env=env,
        architecture=args.architecture,
        num_features=args.num_features,
        context_size=0,
    )

    policy_net = policy_net.to(DEVICE)
    value_net = value_net.to(DEVICE)

    policy_optimizer = tc.optim.AdamW(
        get_weight_decay_param_groups(policy_net, args.adam_wd),
        lr=args.adam_lr,
        eps=args.adam_eps,
    )
    value_optimizer = tc.optim.AdamW(
        get_weight_decay_param_groups(value_net, args.adam_wd),
        lr=args.adam_lr,
        eps=args.adam_eps,
    )

    policy_scheduler = None
    value_scheduler = None

    # load checkpoint, if applicable.
    pol_iters_so_far = 0
    if comm.Get_rank() == ROOT_RANK:
        a = maybe_load_checkpoint(
            checkpoint_dir=args.checkpoint_dir,
            model_name=f"{args.model_name}/policy_net",
            model=policy_net,
            optimizer=policy_optimizer,
            scheduler=policy_scheduler,
            steps=None,
        )

        b = maybe_load_checkpoint(
            checkpoint_dir=args.checkpoint_dir,
            model_name=f"{args.model_name}/value_net",
            model=value_net,
            optimizer=value_optimizer,
            scheduler=value_scheduler,
            steps=None,
        )

        if a != b:
            raise RuntimeError(
                "Policy and value iterates not aligned in latest checkpoint!"
            )
        pol_iters_so_far = a

    # sync state.
    pol_iters_so_far = comm.bcast(pol_iters_so_far, root=ROOT_RANK)
    sync_state(
        model=policy_net,
        optimizer=policy_optimizer,
        scheduler=policy_scheduler,
        comm=comm,
        root=ROOT_RANK,
    )
    sync_state(
        model=value_net,
        optimizer=value_optimizer,
        scheduler=value_scheduler,
        comm=comm,
        root=ROOT_RANK,
    )

    # make callback functions for checkpointing.
    policy_checkpoint_fn = partial(
        save_checkpoint,
        checkpoint_dir=args.checkpoint_dir,
        model_name=f"{args.model_name}/policy_net",
        model=policy_net,
        optimizer=policy_optimizer,
        scheduler=policy_scheduler,
    )

    value_checkpoint_fn = partial(
        save_checkpoint,
        checkpoint_dir=args.checkpoint_dir,
        model_name=f"{args.model_name}/value_net",
        model=value_net,
        optimizer=value_optimizer,
        scheduler=value_scheduler,
    )

    # run it!
    if args.meta_episodes_per_policy_update == -1:
        numer = 240000
        denom = comm.Get_size() * args.num_meta_episodes * args.max_episode_len
        meta_episodes_per_policy_update = numer // denom
    else:
        meta_episodes_per_policy_update = args.meta_episodes_per_policy_update

    training_loop(
        env=env,
        policy_net=policy_net,
        value_net=value_net,
        policy_optimizer=policy_optimizer,
        value_optimizer=value_optimizer,
        policy_scheduler=policy_scheduler,
        value_scheduler=value_scheduler,
        meta_episodes_per_policy_update=meta_episodes_per_policy_update,
        meta_episodes_per_learner_batch=args.meta_episodes_per_learner_batch,
        num_meta_episodes=args.num_meta_episodes,
        ppo_opt_epochs=args.ppo_opt_epochs,
        ppo_clip_param=args.ppo_clip_param,
        ppo_ent_coef=args.ppo_ent_coef,
        discount_gamma=args.discount_gamma,
        gae_lambda=args.gae_lambda,
        standardize_advs=bool(args.standardize_advs),
        max_pol_iters=args.max_pol_iters,
        pol_iters_so_far=pol_iters_so_far,
        policy_checkpoint_fn=policy_checkpoint_fn,
        value_checkpoint_fn=value_checkpoint_fn,
        comm=comm,
        log_wandb=args.log_wandb,
    )


if __name__ == "__main__":
    main()

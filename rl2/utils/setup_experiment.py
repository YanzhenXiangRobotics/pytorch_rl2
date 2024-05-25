from rl2.envs.bandit_env import BanditEnv
from rl2.envs.mdp_env import MDPEnv
from rl2.envs.stackelberg.mat_game_follower_env import (
    MatGameFollowerEnv,
    IteratedMatrixGame,
)
from rl2.envs.stackelberg.drone_game_follower_env import DroneGameFollowerEnv, DroneGame
from rl2.envs.stackelberg.drone_game import DroneGameEnv

from rl2.agents.preprocessing.tabular import (
    MABPreprocessing,
    MDPPreprocessing,
    DGFPreprocessing,
)
from rl2.agents.architectures.gru import GRU
from rl2.agents.architectures.lstm import LSTM
from rl2.agents.architectures.snail import SNAIL
from rl2.agents.architectures.transformer import Transformer
from rl2.agents.heads.policy_heads import LinearPolicyHead
from rl2.agents.heads.value_heads import LinearValueHead
from rl2.agents.integration.policy_net import StatefulPolicyNet
from rl2.agents.integration.value_net import StatefulValueNet

from rl2.utils.constants import DEVICE
from rl2.utils.checkpoint_util import maybe_load_checkpoint


def create_env(config):
    if config.env.name == "bandit":
        return BanditEnv(num_actions=config.bandit.num_actions)
    if config.env.name == "tabular_mdp":
        return MDPEnv(
            num_states=config.mdp.num_states,
            num_actions=config.mdp.num_actions,
            max_episode_length=config.mdp.episode_len,
        )
    if config.env.name == "matrix_game_follower":
        return MatGameFollowerEnv(
            env=IteratedMatrixGame(
                matrix="prisoners_dilemma",
                episode_length=config.matrix_game.episode_len,
                memory=config.matrix_game.memory,
            )
        )
    if config.env.name == "drone_game_follower":
        return DroneGameFollowerEnv(
            env=DroneGame(
                env=DroneGameEnv(
                    width=config.drone_game.width,
                    height=config.drone_game.height,
                    drone_dist=config.drone_game.drone_dist,
                ),
                headless=config.drone_game.headless,
            )
        )
    raise NotImplementedError


def create_preprocessing(env):
    if env.name == "bandit":
        return MABPreprocessing(num_actions=env.num_actions)
    if (env.name == "tabular_mdp") or (env.name == "matrix_game_follower"):
        return MDPPreprocessing(num_states=env.num_states, num_actions=env.num_actions)
    if env.name == "drone_game_follower":
        return DGFPreprocessing(
            num_states=env.num_states,
            dim_states=env.dim_states,
            num_actions=env.num_actions,
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
    env,
    architecture,
    num_features,
    context_size,
):
    preprocessing = create_preprocessing(
        env=env,
    ).to(DEVICE)
    architecture = create_architecture(
        architecture=architecture,
        input_dim=preprocessing.output_dim,
        num_features=num_features,
        context_size=context_size,
    ).to(DEVICE)
    head = create_head(
        head_type=net_type,
        num_features=architecture.output_dim,
        num_actions=env.num_actions,
    ).to(DEVICE)

    if net_type == "policy":
        return StatefulPolicyNet(
            preprocessing=preprocessing, architecture=architecture, policy_head=head
        )
    if net_type == "value":
        return StatefulValueNet(
            preprocessing=preprocessing, architecture=architecture, value_head=head
        )
    raise NotImplementedError


def get_policy_net_for_inference(config, env):
    # create learning system.
    policy_net = create_net(
        net_type="policy",
        env=env,
        architecture=config.model.architecture,
        num_features=config.model.num_features,
        context_size=0,
    )

    policy_net = policy_net.to(DEVICE)

    # load checkpoint, if applicable.
    maybe_load_checkpoint(
        checkpoint_dir=config.model.checkpoint_dir,
        model_name=f"{config.model.model_name}/policy_net",
        model=policy_net,
        optimizer=None,
        scheduler=None,
        steps=None,
    )

    return policy_net

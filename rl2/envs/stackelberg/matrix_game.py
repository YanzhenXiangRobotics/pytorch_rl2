from typing import Union

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

named_matrix_games = {
    "prisoners_dilemma": [
        [[3, 3], [1, 4]],
        [[4, 1], [2, 2]],
    ],
    "stag_hunt": [
        [[4, 4], [1, 3]],
        [[3, 1], [2, 2]],
    ],
    "assurance": [
        [[4, 4], [1, 2]],
        [[2, 1], [3, 3]],
    ],
    "coordination": [
        [[4, 4], [2, 1]],
        [[1, 2], [3, 3]],
    ],
    "mixedharmony": [
        [[4, 4], [3, 1]],
        [[1, 3], [2, 2]],
    ],
    "harmony": [
        [[4, 4], [3, 2]],
        [[2, 3], [1, 1]],
    ],
    "noconflict": [
        [[4, 4], [2, 3]],
        [[3, 2], [1, 1]],
    ],
    "deadlock": [
        [[2, 2], [1, 4]],
        [[4, 1], [3, 3]],
    ],
    "prisoners_delight": [
        [[1, 1], [2, 4]],
        [[4, 2], [3, 3]],
    ],
    "hero": [
        [[1, 1], [3, 4]],
        [[4, 3], [2, 2]],
    ],
    "battle": [
        [[2, 2], [3, 4]],
        [[4, 3], [1, 1]],
    ],
    "chicken": [
        [[3, 3], [2, 4]],
        [[4, 2], [1, 1]],
    ],
}


class IteratedMatrixGame(ParallelEnv):
    """A very basic marix game environment."""

    metadata = {
        "name": "iterated_matrix_game",
    }

    def __init__(
        self,
        matrix: Union[np.ndarray, str] = "prisoners_dilemma",
        episode_length: int = 1,
        memory: int = 0,
        reward_offset: float = -2.5,
    ):
        """Creates a simple matrix game.
        Arguments:

        - matrix: A 3D numpy array of shape (rows, cols, 2) containing the payoff (bi-)matrix. Alternatively, a string can be passed, identifying one of several canonical games. The first dimension corresponds to the action of the first agent, the second to the action of the second agent, and the third to the rewards of the two agents.
        - episode_length: The length of an episode.
        - memory: 0: Empty observation, 1: Previous action of the other agent, 2: Previous action of both agents.
        """
        super().__init__()

        if isinstance(matrix, str):
            matrix = np.array(named_matrix_games[matrix])
        self.matrix = matrix
        self.agents = ["leader", "follower"]
        self.possible_agents = self.agents
        self.memory = memory

        # 0: cooperate, 1: defect
        self.action_spaces = {
            "leader": spaces.Discrete(2),
            "follower": spaces.Discrete(2),
        }
        self.observation_spaces = {
            "leader": spaces.Discrete(memory * 2 + 1),
            "follower": spaces.Discrete(memory * 2 + 1),
        }

        self.episode_length = episode_length
        self.current_step = 0
        self.reward_offset = reward_offset

    def action_space(self, agent: str) -> spaces.Space:
        return self.action_spaces[agent]

    def observation_space(self, agent: str) -> spaces.Space:
        return self.observation_spaces[agent]

    def reset(self):
        self.current_step = 0
        obs = 0
        return {"leader": obs, "follower": obs}

    def step(self, actions):
        self.current_step += 1

        rewards = {
            "leader": self.matrix[actions["leader"]][actions["follower"]][0]
            + self.reward_offset,
            "follower": self.matrix[actions["leader"]][actions["follower"]][1]
            + self.reward_offset,
        }

        if self.memory == 0:
            obs = {"leader": 0, "follower": 0}
        elif self.memory == 1:
            obs = {
                "leader": 1 + actions["follower"],
                "follower": 1 + actions["leader"],
            }
            # 0: first step
            # 1: other agent cooperated
            # 2: other agent defected
        elif self.memory == 2:
            obs = {
                "leader": 1 + actions["leader"] + 2 * actions["follower"],
                "follower": 1 + actions["leader"] + 2 * actions["follower"],
            }
            # 0: first step
            # 1: both agents cooperated
            # 2: leader defected, follower cooperated
            # 3: leader cooperated, follower defected
            # 4: both agents defected

        term = self.current_step >= self.episode_length
        terminated = {"leader": term, "follower": term}
        truncated = {"leader": False, "follower": False}
        info = {"leader": {}, "follower": {}}

        return obs, rewards, terminated, truncated, info


if __name__ == "__main__":
    env = IteratedMatrixGame(matrix="prisoners_dilemma", episode_length=10, memory=2)
    obs = env.reset()

    while True:
        obs, rewards, terminated, truncated, infos = env.step(
            {agent: env.action_space(agent).sample() for agent in env.agents}
        )
        print(obs, rewards)

        if any(terminated.values()) or any(truncated.values()):
            break
    print("Done")

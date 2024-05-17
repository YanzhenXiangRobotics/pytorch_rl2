from __future__ import annotations

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List, Tuple

import time

import numpy as np

from envs.lava_colored import LavaColored

from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal, Wall
from minigrid.minigrid_env import MiniGridEnv

from gymnasium import spaces

from pettingzoo import ParallelEnv

from util.point import Point2D


class DroneGameEnv(MiniGridEnv):
    def __init__(
        self,
        size=22,
        agent_start_pos=None,
        agent_start_dir=0,
        agent_dir_fixed=True,
        agent_view_size=3,
        max_steps: int | None = None,
        drone_options: List = [(15, 3), (15, 8), (15, 13), (15, 18)],
        num_divisions=4,
        drone_cover_size=3,
        **kwargs,
    ):
        if agent_start_pos is None:
            self.agent_start_pos = (
                np.random.randint(1, min(int(0.7 * size), size - 1)),
                np.random.randint(1, size - 1),
            )
        else:
            self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.agent_dir_fixed = agent_dir_fixed
        self.agent_view_size = agent_view_size
        self.drone_options = drone_options
        self.num_divisions = num_divisions
        self.drone_cover_size = drone_cover_size

        mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = 4 * size**2

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            agent_view_size=agent_view_size,
            # Set this to True for maximum speed
            see_through_walls=True,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return "drone game"

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)

        self.grid.wall_rect(0, 0, width, height)

        for j in range(1, self.width - 1):
            self.put_obj(Goal(), self.height - 2, j)

        # if self.agent_start_pos is not None:
        self.agent_pos = self.agent_start_pos
        self.agent_dir = self.agent_start_dir
        # else:
        #     self.place_agent()

        self.mission = "drone game"

    def in_grid(self, point: Point2D, exclude_goal_line=True):
        return point >= Point2D(1, 1) and point <= Point2D(
            self.width - 2 - int(exclude_goal_line),
            self.height - 2,
        )


class DroneGame(ParallelEnv):
    metadata = {
        "name": "drone_game",
    }

    def __init__(
        self, env: DroneGameEnv, headless: bool = False, verbose: bool = False
    ) -> None:
        super().__init__()

        self.env = env
        self.headless = headless
        self.verbose = verbose
        if not headless:
            self.env.render_mode = "human"

        self.agents = ["leader", "follower"]

        agent_view_area = env.agent_view_size * env.agent_view_size
        # grid_area = (env.width - 2) * (env.height - 2) # The outer are walls so -2

        # leader action: which of prescribed places to place drone
        # follower action: fwd(0), fwd-left(1), fwd-right(2)
        self.action_spaces = {
            "leader": spaces.Discrete(len(self.env.drone_options)),
            "follower": spaces.Discrete(3),
        }
        # leader observation: which division does the follower lie in
        # follower observation: wall occupancy in its local view size
        self.observation_spaces = {
            "leader": spaces.MultiBinary(self.env.num_divisions),
            "follower": spaces.MultiDiscrete(
                [self.env.height, *([4] * agent_view_area)]
            ),
        }

        self.drones = []

    def action_space(self, agent: str) -> spaces.Space:
        return self.action_spaces[agent]

    def observation_space(self, agent: str) -> spaces.Space:
        return self.observation_spaces[agent]

    def reset(self):
        self.env.reset()
        return {
            "leader": np.zeros(self.env.num_divisions),
            "follower": np.zeros(
                1 + self.env.agent_view_size * self.env.agent_view_size
            ),
        }

    def step(self, actions):
        self.transition_drones()
        self.leader_act(actions["leader"])
        self.follower_act(actions["follower"])

        observations, rewards = {}, {}
        observations["leader"] = self.get_leader_observation()
        observations["follower"] = self.get_follower_observation()
        rewards["leader"] = self.get_leader_reward()
        rewards["follower"] = -rewards["leader"]

        terminated = (self.env.agent_pos[0] == self.env.height - 2) or isinstance(
            self.env.grid.get(self.env.agent_pos[0], self.env.agent_pos[1]), LavaColored
        )
        terminated = {"leader": terminated, "follower": terminated}
        truncated = {"leader": False, "follower": False}
        info = {"leader": {}, "follower": {}}

        if self.verbose:
            print(f"\n STEP: {self.env.step_count}\n")
            print(f"\nterminated: {terminated}")
            print(f"Leader takes action {actions['leader']}")
            print(f"leader observation: {observations['leader']}")
            print(f"\nFollower takes action {actions['follower']}")
            print(f"follower observation: {observations['follower']}")
            print(f"follower observation binary: {observations['follower']}")
            print(f"follower reward: {rewards['follower']}")
            print(f"leader reward: {rewards['leader']}")

        if not self.headless:
            self.env.render()
            time.sleep(1.0)

        return observations, rewards, terminated, truncated, info

    def leader_act(self, action):
        drone_place = self.env.drone_options[action]
        self.drones.append(Drone(env=self.env, radius=1, center=drone_place))

    def transition_drones(self):
        if len(self.drones) >= 4:
            dead_drone = self.drones.pop(0)
            dead_drone.undo_lava()
            del dead_drone

        for drone in self.drones:
            drone.undo_lava()
        for drone in self.drones:
            drone.brownian_motion()
        for drone in self.drones:
            drone.set_lava()

    def get_leader_observation(self):
        observation = np.zeros(self.env.num_divisions)

        for drone in self.drones:
            dists = [
                drone.center.euclidean_distance(Point2D(i, j))
                for i, j in self.env.drone_options
            ]
            index = np.argmin(dists)
            observation[index] = 1

        return observation

    def get_leader_reward(self):
        if isinstance(
            self.env.grid.get(self.env.agent_pos[0], self.env.agent_pos[1]), LavaColored
        ):
            return 5 * self.env.height

        return 0

    def follower_act(self, action):
        self.env.render_mode = None
        match action:
            case 0:  # fwd
                self.env.step(self.env.actions.forward)
            case 1:  # left
                self.env.step(self.env.actions.left)
                self.env.step(self.env.actions.forward)
                self.env.step(self.env.actions.right)
                self.env.step(self.env.actions.forward)
                self.env.step_count -= 3
            case 2:  # right
                self.env.step(self.env.actions.right)
                self.env.step(self.env.actions.forward)
                self.env.step(self.env.actions.left)
                self.env.step(self.env.actions.forward)
                self.env.step_count -= 3
        self.env.render_mode = "human" if not self.headless else None

    def get_follower_observation(self):
        topX, topY, botX, botY = self.env.get_view_exts()

        observation = np.zeros((self.env.agent_view_size, self.env.agent_view_size))
        for i in range(topX, botX):
            for j in range(topY, botY):
                i_local, j_local = self.env.relative_coords(i, j)

                if not self.env.in_grid(Point2D(i, j)):
                    observation[i_local, j_local] = 1
                elif isinstance(self.env.grid.get(i, j), LavaColored):
                    observation[i_local, j_local] = 2
                elif isinstance(self.env.grid.get(i, j), Wall):
                    observation[i_local, j_local] = 3

        return np.insert(observation.flatten(), 0, self.env.agent_pos[1])


class Drone:
    def __init__(self, env: DroneGameEnv, center: Tuple[int, int], radius=1) -> None:
        self.radius = radius
        # self.to_death = lifespan
        self.center = Point2D(*center)
        self.env = env
        self.age = 0
        self.set_lava()

    def in_grid(self, point: Point2D = None):
        point = point or self.center
        return (
            self.env.in_grid(point + Point2D(self.radius, self.radius))
            and self.env.in_grid(point + Point2D(self.radius, -self.radius))
            and self.env.in_grid(point + Point2D(-self.radius, self.radius))
            and self.env.in_grid(point + Point2D(-self.radius, -self.radius))
        )

    def fill_body(self, type):
        for i in range(
            self.center.x - self.radius,
            self.center.x + self.radius + 1,
        ):
            for j in range(
                self.center.y - self.radius,
                self.center.y + self.radius + 1,
            ):
                if self.env.in_grid(Point2D(i, j)):
                    if type is None:
                        self.env.grid.set(i, j, None)
                    else:
                        self.env.put_obj(type, i, j)

    def set_lava(self):
        self.fill_body(LavaColored(self.age <= 0))

    def undo_lava(self):
        self.fill_body(None)

    def brownian_motion(self):
        self.age += 1
        dirs = [(0, 1), (0, -1), (-1, 0), (1, 0)]
        dirs = list(filter(lambda x: self.in_grid(self.center + Point2D(*x)), dirs))

        index = np.random.randint(len(dirs))
        move = Point2D(*dirs[index])
        next_center = self.center + move

        self.center = next_center


if __name__ == "__main__":
    # env = DroneGameEnv(agent_start_pos=(3, 10), agent_dir_fixed=True)
    env = DroneGameEnv(agent_start_pos=(1, 10), agent_dir_fixed=True, agent_start_dir=0)
    env = DroneGame(env=env, verbose=True)

    follower_action_seq = [0, 0, 0, 0, 0, 2, 2, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]

    i = 0
    env.env.reset()
    while True:
        actions = {}
        # actions["follower"] = env.action_space("follower").sample()
        if i < len(follower_action_seq):
            actions["follower"] = follower_action_seq[i]
        else:
            actions["follower"] = 0
        actions["leader"] = i % 4
        observation, reward, terminated, _, _ = env.step(actions)
        if terminated["follower"]:
            break
        i += 1

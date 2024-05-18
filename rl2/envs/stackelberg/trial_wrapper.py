from pettingzoo.utils.wrappers import BaseParallelWrapper

from rl2.envs.stackelberg.matrix_game import IteratedMatrixGame


class TrialWrapper(BaseParallelWrapper):
    def __init__(
        self,
        env,
        num_episodes: int,
        zero_leader_reward: bool = True,
        zero_follower_reward: bool = False,
    ):
        assert num_episodes >= 2, "num_episodes must be greater than or equal to 2"

        super().__init__(env)
        self.env = env
        self.num_episodes = num_episodes
        self.zero_leader_reward = zero_leader_reward
        self.zero_follower_reward = zero_follower_reward

        self.current_episode = 0

    # Start a fresh episode inside this trial
    # A trial is a sequence of episodes (as in RL^2)
    def _inner_reset(self):
        self.current_episode += 1
        obs = self.env.reset()
        return obs

    def reset(self):
        self.current_episode = 0
        return self._inner_reset()

    def step(self, actions):
        obs, reward, term, trunc, info = self.env.step(actions)

        if self.zero_leader_reward and self.current_episode < self.num_episodes:
            reward["leader"] = 0.0

        if self.zero_follower_reward and self.current_episode < self.num_episodes:
            reward["follower"] = 0.0

        # Only set terminated to True if the last episode is terminated
        terminated = False
        if any(term.values()) or any(trunc.values()):
            if self.current_episode == self.num_episodes:
                terminated = True
            else:
                obs = self._inner_reset()

        term = {"follower": terminated, "leader": terminated}
        trunc = {"follower": False, "leader": False}

        return obs, reward, term, trunc, info

    def render():
        pass

    def close():
        pass


if __name__ == "__main__":
    env = IteratedMatrixGame(matrix="prisoners_dilemma", episode_length=10, memory=2)
    env = TrialWrapper(env, num_episodes=2)
    obs = env.reset()

    i = 1

    while True:
        i += 1
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        new_obs, rewards, terminated, truncated, infos = env.step(actions)

        print("STEP")
        print(obs)
        print(actions)
        print(rewards)
        print(new_obs)

        if any(terminated.values()) or any(truncated.values()):
            break

        obs = new_obs
    print("Done", i)

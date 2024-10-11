import gym
import numpy as np


class ChannelFirstEnv(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_space = self.observation_space
        obs_shape = obs_space.shape[-1:] + obs_space.shape[:2]
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=obs_shape, dtype=np.uint8
        )

    def _permute_orientation(self, observation):
        # permute [H, W, C] array to [C, H, W] tensor
        observation = np.transpose(observation, (2, 0, 1))
        return observation

    def observation(self, observation):
        observation = self._permute_orientation(observation)
        return observation


class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip, test=False):
        super().__init__(env)
        self._skip = skip
        self.test = test

    def step(self, action):
        total_reward = 0.0
        for i in range(self._skip):
            output = self.env.step(action)
            obs, reward, terminated, truncated, info = output
            done = terminated or truncated
            total_reward += reward
            if done:
                break
        if self.test:
            return obs, total_reward, terminated, truncated, info
        return obs, total_reward, done, info


class PixelNormalization(gym.Wrapper):
    def __init__(self, env, test=False):
        super().__init__(env)
        self.test = test

    def _pixel_normalization(self, obs):
        if len(obs) == 2:
            obs = obs[0]
        return obs / 255.0 - 0.5

    def step(self, action):
        output = self.env.step(action)
        if self.test:
            obs, reward, terminated, truncated, info = output
            return self._pixel_normalization(obs), reward, terminated, truncated, info
        obs, reward, done, info = output
        return self._pixel_normalization(obs), reward, done, info

    def reset(self):
        obs = self.env.reset()
        return self._pixel_normalization(obs)

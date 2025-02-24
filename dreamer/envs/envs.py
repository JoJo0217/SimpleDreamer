import gym
import dmc2gym

from dreamer.envs.wrappers import *


def make_dmc_env(
    domain_name,
    task_name,
    seed,
    visualize_reward,
    from_pixels,
    height,
    width,
    frame_skip,
    pixel_norm=True,
):
    env = dmc2gym.make(
        domain_name=domain_name,
        task_name=task_name,
        seed=seed,
        visualize_reward=visualize_reward,
        from_pixels=from_pixels,
        height=height,
        width=width,
        frame_skip=frame_skip,
    )
    if pixel_norm:
        env = PixelNormalization(env)
    return env


def make_atari_env(task_name, skip_frame, width, height, seed, pixel_norm=True, test=False):
    if test:
        env = gym.make(task_name, render_mode="rgb_array")
    else:
        env = gym.make(task_name)
    env = gym.wrappers.ResizeObservation(env, (height, width))
    env = ChannelFirstEnv(env)
    env = SkipFrame(env, skip_frame, test)
    if pixel_norm:
        env = PixelNormalization(env, test)
    return env


def get_env_infos(env):
    obs_shape = env.observation_space.shape
    if isinstance(env.action_space, gym.spaces.Discrete):
        discrete_action_bool = True
        action_size = env.action_space.n
    elif isinstance(env.action_space, gym.spaces.Box):
        discrete_action_bool = False
        action_size = env.action_space.shape[0]
    else:
        raise Exception
    return obs_shape, discrete_action_bool, action_size

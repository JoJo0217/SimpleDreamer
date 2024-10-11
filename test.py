import argparse
import torch
from dreamer.utils.utils import load_config, get_base_directory
from dreamer.envs.envs import make_dmc_env, make_atari_env, get_env_infos
from dreamer.modules.model import RSSM
from dreamer.modules.encoder import Encoder
from dreamer.modules.actor import Actor
import gym

env = make_atari_env


def main(config_file):
    config = load_config(config_file)

    if config.environment.benchmark == "atari":
        env = make_atari_env(
            task_name=config.environment.task_name,
            seed=config.environment.seed,
            height=config.environment.height,
            width=config.environment.width,
            skip_frame=config.environment.frame_skip,
            pixel_norm=config.environment.pixel_norm,
            test=True,
        )
    elif config.environment.benchmark == "dmc":
        env = make_dmc_env(
            domain_name=config.environment.domain_name,
            task_name=config.environment.task_name,
            seed=config.environment.seed,
            visualize_reward=config.environment.visualize_reward,
            from_pixels=config.environment.from_pixels,
            height=config.environment.height,
            width=config.environment.width,
            frame_skip=config.environment.frame_skip,
            pixel_norm=config.environment.pixel_norm,
        )
    obs_shape, discrete_action_bool, action_size = get_env_infos(env)

    device = config.operation.device

    encoder = Encoder(obs_shape, config).to(device)
    rssm = RSSM(action_size, config).to(device)
    actor = Actor(discrete_action_bool, action_size, config).to(device)

    encoder.load_state_dict(torch.load('encoder.pth'))
    rssm.load_state_dict(torch.load('rssm.pth'))
    actor.load_state_dict(torch.load('actor.pth'))

    recording = True

    if recording:
        env = gym.wrappers.RecordVideo(env, video_folder="./video",
                                       episode_trigger=lambda x: x % 1 == 0)

    posterior, deterministic = rssm.recurrent_model_input_init(1)
    action = torch.zeros(1, action_size).to(device)

    observation = env.reset()
    embedded_observation = encoder(
        torch.from_numpy(observation).float().to(device)
    )
    score = 0
    done = False

    while not done:
        with torch.no_grad():
            deterministic = rssm.recurrent_model(
                posterior, action, deterministic
            )
            embedded_observation = embedded_observation.reshape(1, -1)
            _, posterior = rssm.representation_model(
                embedded_observation, deterministic
            )
            action = actor(posterior, deterministic).detach()

            if discrete_action_bool:
                buffer_action = action.cpu().numpy()
                env_action = buffer_action.argmax()
            else:
                buffer_action = action.cpu().numpy()[0]
                env_action = buffer_action

            output = env.step(env_action)
            next_observation, reward, terminate, truncated, info = output
            done = terminate or truncated
            score += reward
            embedded_observation = encoder(
                torch.from_numpy(next_observation).float().to(device)
            )
            observation = next_observation
    print(score)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="racing.yml",
        help="config file to run(default: dmc-walker-walk.yml)",
    )
    main(parser.parse_args().config)

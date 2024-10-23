from copy import deepcopy
from functools import partial
import gymnasium as gym
import numpy as np
from numba import njit
import matplotlib.pyplot as plt
from gymnasium.experimental.wrappers import FrameStackObservationV0
from gymnasium.envs.registration import register

from planning.planner_wrapper import BasePlannerEnvWrapper, OpponentAutopilotEnvWrapper
from racemop.env_wrapper import SliceStackedObs
from racemop.agents import agents
from racemop import env_wrapper

register(id='f110-v1.0', entry_point='f110_gym.envs:F110Env')


@njit
def adaptive_centered_median_filter_with_downsampling(signal, angle_data, high_res_angle, high_res_window,
                                                      low_res_window, high_res_downsampling_factor,
                                                      low_res_downsampling_factor):
    """Apply an adaptive centered median filter with downsampling to the input signal.

    Args:
        signal (np.ndarray): Input signal to be filtered
        angle_data (np.ndarray): Angle data corresponding to the signal
        high_res_angle (float): Angle threshold for high-resolution filtering
        high_res_window (int): Window size for high-resolution filtering
        low_res_window (int): Window size for low-resolution filtering
        high_res_downsampling_factor (int): Downsampling factor for high-resolution filtering
        low_res_downsampling_factor (int): Downsampling factor for low-resolution filtering
    Returns:
        np.ndarray: Filtered signal
        np.ndarray: Filtered angle data
    """
    filtered_signal = []
    fov = []
    for i in range(len(signal)):
        # Determine the window size and downsampling factor based on angle
        if abs(angle_data[i]) <= high_res_angle:
            window_size = high_res_window
            downsampling_factor = high_res_downsampling_factor
        else:
            window_size = low_res_window
            downsampling_factor = low_res_downsampling_factor

        # Apply downsampling
        if i % downsampling_factor == 0:
            # Calculate the window range
            half_window = window_size // 2
            start_index = max(i - half_window, 0)
            end_index = min(i + half_window + 1, len(signal))

            # Apply median filter for this window
            window = signal[start_index:end_index]
            filtered_signal.append(np.median(window))
            fov.append(angle_data[i])

    return np.array(filtered_signal), np.array(fov)


@njit
def _norm_scan(scan: dict):
    """Normalize the scan data to [0, 1].

    Args:
        scan (np.ndarray): Input scan data
    Returns:
        np.ndarray: Normalized scan data
    """
    min_scan = 0.0
    max_scan = 10.0
    scan = np.clip(scan, min_scan, max_scan)
    # normalize to [0, 1]
    scan = (scan - min_scan) / (max_scan - min_scan)
    return scan


def norm_scan(obs, config):
    """Normalize the scan data to [0, 1].

    Args:
        obs (dict): Observation dictionary
        config: Config object
    Returns:
        dict: Normalized observation dictionary
    """
    data = deepcopy(obs['aaa_scans'][0])

    if config.env.wrapper.lidar_downsample_filter:
        # Fov is 270 deg with 1080 points
        fov = np.linspace(-135, 135, 1080)
        # Parameters for adaptive filtering and downsampling
        high_res_angle = 45  # Degrees, adjust as needed
        high_res_window = 3  # Window size for high-resolution areas
        low_res_window = 3  # Window size for low-resolution areas
        high_res_downsampling_factor = 1  # Downsampling factor for high-resolution areas
        low_res_downsampling_factor = 1  # Downsampling factor for low-resolution areas

        # Apply adaptive centered median filter with downsampling
        data_filtered, fov_filtered = adaptive_centered_median_filter_with_downsampling(
            data, fov, high_res_angle, high_res_window, low_res_window, high_res_downsampling_factor,
            low_res_downsampling_factor
        )
        data = data_filtered
    data = _norm_scan(data)

    obs['aaa_scans'] = np.expand_dims(data, axis=0)

    do_plot = False
    if do_plot:
        fig = plt.gcf()
        if not fig.axes:
            ax1 = fig.add_subplot(1, 2, 1)
            ax2 = fig.add_subplot(1, 2, 2)
        else:
            ax1, ax2 = fig.axes
        ax1.cla()
        ax2.cla()
        ax1.grid(True)
        ax2.grid(True)
        ax1.set_ylim(-0.1, 1.1)
        ax2.set_ylim(-0.1, 1.1)
        ax1.scatter(fov, obs['aaa_scans'][0], s=3, c='b')
        ax2.scatter(fov_filtered, data_filtered, s=3, c='g')
        plt.pause(1e-5)
        print(len(obs['aaa_scans'][0]))
        print(len(data_filtered))
    return obs


def make_env(config, map_names, seed):
    """Create a gym environment with the given config and map names.

    Args:
        config: Config object
        map_names: List of map names to be used in the environment
        seed: Seed for the environment
    Returns:
        A function that creates a gym environment
    """

    map_names = [map_names] if type(map_names) is not list else map_names

    def thunk():
        env = gym.make(
            'f110_gym:f110-v1.0',
            config=config,
            map_names=map_names,
            seed=seed,
            render_mode=config.sim.render_mode if config.render else None
        )
        if config.sim.n_agents > 1:
            env = OpponentAutopilotEnvWrapper(env, config)

        # Planner wrapper that includes the planner's action in the observation
        env = BasePlannerEnvWrapper(env, config)

        # Remove unnecessary keys from observation, i.e., filter_keys are kept
        env = gym.wrappers.FilterObservation(env, filter_keys=config.rl.obs_keys)

        # Clip and center scans, use partial to wrap norms_scan with config
        transform_func = partial(norm_scan, config=config)
        env = gym.wrappers.TransformObservation(env, transform_func)

        if config.env.frame_cat.use:
            env = FrameStackObservationV0(env, config.env.frame_cat.n)
            env = SliceStackedObs(env, config.env.frame_cat.skip_slicing)
        return env

    return thunk


def make_multi_env(config, make_functions: list, daemon=True):
    """Create a multi-environment with the given config and make functions.

    Args:
        config: Config object
        make_functions: List of functions that create gym environments
        daemon: Whether to run the environments as daemons
    Returns:
        A multi-environment with the given config and make functions
    """
    if config.async_env:
        return gym.vector.AsyncVectorEnv(make_functions, daemon=daemon, shared_memory=True)
    else:
        return gym.vector.SyncVectorEnv(make_functions)


def make_wrapped_envs(config, envs, eval_mode=False):
    """Apply wrappers to the environment.

    Args:
        config: Config object
        envs: List of gym environments
        eval_mode: Evaluation mode so that no reward normalization is applied

    Returns:
        A gym environment with the given config and map names
    """

    if config.env.wrapper.normalize_obs:
        envs = env_wrapper.NormalizeObservation(envs, exclude_keys=['action_planner', 'aaa_scans'])

    if not eval_mode and config.env.wrapper.normalize_rew:
        envs = gym.wrappers.TransformReward(
            envs, lambda reward: np.clip(reward, config.env.wrapper.clip_rew[0], config.env.wrapper.clip_rew[1])
        )
        envs = env_wrapper.NormalizeReward(envs, gamma=config.rl.gamma)
    return envs


def make_agent(config, action_space, obs_space, device):
    """Create an agent with the given config, action space, observation space, and device.

    Args:
        config: Config object
        action_space: Action space of the environment
        obs_space: Observation space of the environment
        device: Device to run the agent on
    Returns:
        An agent with the given config, action space, observation space, and device
    """

    agent = agents.ResidualAgent(config, action_space, obs_space).to(device)
    next_done = False

    return agent, next_done


def make_training_envs(config):
    """Create training environments with the given config.

    Args:
        config: Config object
    Returns:
        Training environments with the given config
    """
    assert config.rl.num_envs % len(
        config.maps.maps_train) == 0, "Number of envs must be divisible by number of maps."
    maps = list(config.maps.maps_train) * int(config.rl.num_envs / len(config.maps.maps_train))
    maps.sort()
    envs = make_multi_env(config, ([make_env(config, maps[i], seed=config.seed + i) for i in range(len(maps))]))
    envs = make_wrapped_envs(config, envs, eval_mode=False)
    envs.map_names = maps
    action_space = envs.get_wrapper_attr('single_action_space')
    obs_space = envs.get_wrapper_attr('single_observation_space')
    return envs, action_space, obs_space


def make_eval_envs(config):
    """Create evaluation environments with the given config.

    Args:
        config: Config object
    Returns:
        Evaluation environments with the given config
    """
    maps_train = [[m, 'train', ] for m in config.maps.maps_train]
    maps_test = [[m, 'test', ] for m in config.maps.maps_test]
    maps = (maps_train + maps_test) * config.eval_n_parallel_envs
    maps.sort()

    envs = make_multi_env(config, ([make_env(config, maps[i][0], seed=config.seed + i) for i in range(len(maps))]))
    envs = make_wrapped_envs(config, envs, eval_mode=True)
    envs.map_names = [m[0] for m in maps]
    envs.map_types = [m[1] for m in maps]
    action_space = envs.get_wrapper_attr('single_action_space')
    obs_space = envs.get_wrapper_attr('single_observation_space')
    return envs, action_space, obs_space

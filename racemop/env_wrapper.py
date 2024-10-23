import math
from copy import deepcopy
import gymnasium as gym
import numpy as np
from typing import Union, Dict
from functools import partial
import wandb
from gymnasium.experimental.vector.utils import (
    batch_space,
)
from gymnasium.experimental.wrappers import FrameStackObservationV0


class RunningMean:
    """Calculates the running mean of a list of values."""
    def __init__(self, window_size=10):
        self.window_size = window_size
        # init with negative values to avoid bias
        self.values = [-1e6] * window_size

    def update(self, value):
        self.values.append(value)
        if len(self.values) > self.window_size:
            self.values.pop(0)

    def calculate_mean(self):

        if not self.values:
            return None
        return sum(self.values) / len(self.values)

    def step(self, value):
        self.update(value)
        return self.calculate_mean()


class RewardLogger:
    """Logs the rewards of the environment."""
    def __init__(self, config):
        self.config = config
        self.rewards = None
        self.running_mean_rewards = None
        self.running_mean_rewards_collisions = None

    def __call__(self, step, infos):
        if self.rewards is None:
            self.rewards = {k: np.zeros((self.config.rl.num_steps, self.config.rl.num_envs)) for k in infos.keys() if
                            ('reward' in k and k[0] != '_')}
            self.running_mean_rewards = RunningMean(window_size=5)
            self.running_mean_rewards_collisions = RunningMean(window_size=5)

        if 'final_info' in infos.keys():
            for k in self.rewards.keys():
                infos[k][infos['_final_info']] = infos['final_info'][infos['_final_info']][0][k]

        for k in self.rewards.keys():
            self.rewards[k][step] = infos[k]

        if step == self.config.rl.num_steps - 1:
            reward_log = dict()
            for k in self.rewards.keys():
                reward_log[f'rewards/{k}_mean'] = self.rewards[k].mean()
                if k == 'reward':
                    reward_log[f'rewards/{k}_std'] = self.rewards[k].std()
                    reward_log[f'rewards/{k}_min'] = self.rewards[k].min()
                    reward_log[f'rewards/{k}_max'] = self.rewards[k].max()
                    self.running_mean_rewards.update(self.rewards[k].sum())
                if k == 'reward_collision':
                    self.running_mean_rewards_collisions.update(self.rewards[k].mean())
            wandb.log(reward_log)


def clip_over_list_func(config, clip_obs, exclude_keys=None):
    """Clips the observations of the environment."""
    def _clip_over_list(data, clip_obs, exclude_keys=None):
        out = {}
        for key, value in data.items():
            if isinstance(value, dict):
                subout = {}
                for subkey, subvalue in value.items():
                    if subkey not in exclude_keys:
                        subout[subkey] = np.clip(
                            subvalue,
                            clip_obs[0],
                            clip_obs[1]
                        )
                    else:
                        subout[subkey] = subvalue  # .astype(np.float32)
                out[key] = subout
            else:
                if key not in exclude_keys:
                    out[key] = np.clip(
                        value,
                        clip_obs[0],
                        clip_obs[1]
                    )
                else:
                    out[key] = value  # .astype(np.float32)
        return out

    return partial(_clip_over_list, clip_obs=config.env.wrapper.clip_obs, exclude_keys=exclude_keys)


class NormalizeObservation(gym.wrappers.NormalizeObservation):
    """Wrapper that can block the update of the running mean and variance of the observations.

    Attributes:
        _block_update_obs (bool): Variable to block the update of the running mean and variance of the observations.
    """
    def __init__(self, env: gym.Env, epsilon: float = 1e-8, exclude_keys=['']):
        gym.utils.RecordConstructorArgs.__init__(self, epsilon=epsilon)
        gym.Wrapper.__init__(self, env)
        try:
            self.num_envs = self.get_wrapper_attr("num_envs")
            self.is_vector_env = self.get_wrapper_attr("is_vector_env")
        except AttributeError:
            self.num_envs = 1
            self.is_vector_env = False

        self.epsilon = epsilon
        self.num_envs = env.get_wrapper_attr("num_envs")
        self.is_vector_env = env.get_wrapper_attr("is_vector_env")
        self.is_dict_space = isinstance(env.get_wrapper_attr('single_observation_space'), gym.spaces.dict.Dict)

        self.exclude_keys = exclude_keys
        self.obs_rms: Union[
            gym.wrappers.normalize.RunningMeanStd, Dict[str, gym.wrappers.normalize.RunningMeanStd], None] = None
        if self.is_vector_env:
            if self.is_dict_space:
                self.obs_rms = {}
                for key, space in env.get_wrapper_attr('single_observation_space').items():
                    if isinstance(space, gym.spaces.dict.Dict):
                        rms = {}
                        for subkey, subspace in space.items():
                            if subkey not in exclude_keys:
                                rms[subkey] = gym.wrappers.normalize.RunningMeanStd(shape=subspace.shape)
                        self.obs_rms[key] = rms
                    else:
                        if key not in exclude_keys:
                            self.obs_rms[key] = gym.wrappers.normalize.RunningMeanStd(shape=space.shape)

            else:
                self.obs_rms = gym.wrappers.normalize.RunningMeanStd(
                    shape=env.get_wrapper_attr('single_observation_space').shape)
        else:
            self.obs_rms = gym.wrappers.normalize.RunningMeanStd(
                shape=env.get_wrapper_attr('single_observation_space').shape)
        self._block_update_obs = False

    @property
    def block_update_obs(self):
        return self._block_update_obs

    def set_block_update_obs(self, value):
        """Blocks the update of the running mean and variance of the observations.

        Note: Property is not correctly used. This is a workaround so that this function is exposed in wrapped envs.
        """
        self._block_update_obs = value

    def get_obs_rms(self):
        return deepcopy(self.obs_rms)

    def set_obs_rms(self, value):
        """Sets the running mean and variance of the observations."""
        self.obs_rms = value

    def normalize(self, obs):
        """Normalises the observation using the running mean and variance of the observations."""
        if self.is_dict_space:
            out = {}
            for key, value in obs.items():
                if isinstance(value, dict):
                    subout = {}
                    sub_rms = self.obs_rms[key]
                    for subkey, subvalue in value.items():
                        if subkey not in self.exclude_keys:
                            if not self._block_update_obs:
                                sub_rms[subkey].update(subvalue)
                            subout[subkey] = ((subvalue - sub_rms[subkey].mean) / np.sqrt(
                                sub_rms[subkey].var + self.epsilon)).astype(np.float32)
                        else:
                            subout[subkey] = subvalue.astype(np.float32)
                    out[key] = subout
                else:
                    if key not in self.exclude_keys:
                        if not self._block_update_obs:
                            self.obs_rms[key].update(value)
                        out[key] = ((value - self.obs_rms[key].mean) / np.sqrt(
                            self.obs_rms[key].var + self.epsilon)).astype(np.float32)
                    else:
                        out[key] = value.astype(np.float32)
        else:
            if not self._block_update_obs:
                self.obs_rms.update(obs)
            out = ((obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon)).astype(np.float32)
        return out


class NormalizeReward(gym.wrappers.NormalizeReward):
    """Wrapper that can block the update of the running mean and variance of the rewards.

    Attributes:
        _block_update_rew (bool): Variable to block the update of the running mean and variance of the rewards.
    """

    def __init__(self, *args, **kwargs):
        """Initialises the wrapper."""
        super().__init__(*args, **kwargs)
        self._block_update_rew = False

    @property
    def block_update_rew(self):
        """Variable to block the update of the running mean and variance of the rewards."""
        return self._block_update_rew

    def set_block_update_rew(self, value):
        """Blocks the update of the running mean and variance of the rewards.

        Note: Property is not correctly used. This is a workaround so that this function is exposed in wrapped envs.
        """
        self._block_update_rew = value

    def get_rew_rms(self):
        """Returns the running mean and variance of the rewards."""
        return deepcopy(self.get_wrapper_attr('return_rms'))

    def set_rew_rms(self, value):
        """Sets the running mean and variance of the rewards."""
        self.return_rms = value

    def normalize(self, rews):
        """Returns the normalised reward if not blocked."""
        if not self._block_update_rew:
            self.get_wrapper_attr('return_rms').update(self.returns)
        return rews / np.sqrt(self.get_wrapper_attr('return_rms').var + self.epsilon)


class SliceStackedObs(gym.ObservationWrapper):
    """Wrapper that slices the stacked observations.

    Attributes:
        skip_slicing (int): Number of observations to skip.
        stack_size (int): Size of the stack
    """
    def __init__(self, env: gym.Env, skip_slicing):
        gym.ObservationWrapper.__init__(self, env)
        self.skip_slicing = skip_slicing
        self.stack_size = math.ceil(env.get_wrapper_attr('stack_size') / (self.skip_slicing + 1))
        # Important: Must be used directly after FrameStackObservationV0
        assert isinstance(env, FrameStackObservationV0)
        self.observation_space = batch_space(env.env.observation_space, n=self.stack_size)

    def observation(self, observation):
        # observation['aaa_scans'].shape[0] - 1 means that he last scan is always included as an index
        slicing = np.arange(observation['aaa_scans'].shape[0] - 1, -1, -(self.skip_slicing + 1))
        # flip the array to get the last scan first
        slicing = np.flip(slicing)
        return {k: v[slicing, :] for k, v in observation.items()}

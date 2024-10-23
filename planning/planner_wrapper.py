from abc import ABC
from copy import deepcopy

import gymnasium as gym
import numpy as np

from planning.dummy import DummyPlanner
from planning.ftg.disparity_extender_planner import DisparityExtenderPlanner
from planning.ftg.ftg_planner import FTGPlanner
from planning.ftg.ftg_plus_planner import FTGPlusPlanner
from planning.potential_field.potential_fields_planner import PotentialFieldsPlanner
from planning.pure_pursuit.pure_pursuit import PurePursuitPlanner


class GymActionObservationWrapper(gym.Wrapper):
    """A wrapper that modifies the action and observation space of the environment."""

    def __init__(self, env):
        super().__init__(env)

    def observation(self, observation):
        """Returns a modified observation."""
        raise NotImplementedError

    def action(self, action):
        """Returns a modified action before :meth:`env.step` is called."""

        raise NotImplementedError

    def reverse_action(self, action):
        """Returns a reversed ``action``."""
        raise NotImplementedError

    def step(self, action):
        """Returns a modified observation using :meth:`self.observation` after calling :meth:`env.step`."""
        observation, reward, terminated, truncated, info = self.env.step(self.action(action))
        observation_ = self.observation(observation)
        info['obs_'] = deepcopy(observation_)
        return observation_, reward, terminated, truncated, info

    def reset(self, **kwargs):
        """Resets the environment, returning a modified observation using :meth:`self.observation`."""
        observation, info = self.env.reset(**kwargs)
        observation_ = self.observation(observation)
        info['obs_'] = deepcopy(observation_)
        return observation_, info


class OpponentAutopilotEnvWrapper(GymActionObservationWrapper, ABC):
    def __init__(self, env, config):
        super().__init__(env)
        self.config = config
        self.n_opponents = self.unwrapped.num_agents - 1
        self.autopilot_type = config.planner.opponents.params.type
        self.wpts_type = config.planner.opponents.params.wpts_type
        if self.wpts_type == 'race':
            track = deepcopy(self.unwrapped.racing_line)
            self.waypoints = track.waypoints
        elif self.wpts_type == 'center':
            track = deepcopy(self.unwrapped.center_line)
            self.waypoints = track.waypoints

        config_planner = deepcopy(self.config.planner.opponents.params)
        config_planner['vgain'] = float(np.random.choice(config_planner['vgain']))

        self.autopilots = [
            make_planner(config_planner, self.config.sim.vehicle_params) for _ in range(self.n_opponents)
        ]

        # Purge action and observation space of other agents.py
        observation_space_ = gym.spaces.Dict()
        for k, v in self.get_wrapper_attr('observation_space').items():
            space_class = type(v)
            observation_space_[k] = space_class(v.low[0][np.newaxis], v.high[0][np.newaxis], (1, *v.shape[1:]), v.dtype)
        self.observation_space = observation_space_
        self.autopilot_observation = None

    def observation(self, observation):
        """Returns a modified observation."""
        self.autopilot_observation = deepcopy(observation)
        self.autopilot_observation = [
            {k: np.expand_dims(v[i + 1], axis=0) for k, v in self.autopilot_observation.items() if 1 < len(v)}
            for i in range(self.n_opponents)]
        return {k: np.expand_dims(v[0], axis=0) for k, v in observation.items()}

    def action(self, action):
        """Returns a modified action before :meth:`env.step` is called."""
        autopilot_actions = np.vstack(
            [self.autopilots[i].plan(self.autopilot_observation[i], self.waypoints[:, [0, 1, 2, 4]]) for
             i in range(self.n_opponents)]
        )
        # clip to action space
        autopilot_actions = np.clip(
            autopilot_actions,
            a_min=self.action_space.low,
            a_max=self.action_space.high
        )
        actions = np.vstack((action, autopilot_actions))
        return actions

    def reset(self, *args, **kwargs):
        """Calls :meth:`env.reset` and renders the environment."""
        out = super().reset(*args, **kwargs)

        # update the config
        # OmegaConf.update(config, 'log_dir', HydraConfig.get().run.dir, force_add=True)

        config_planner = deepcopy(self.config.planner.opponents.params)
        config_planner['vgain'] = float(np.random.choice(config_planner['vgain']))

        self.autopilots = [make_planner(config_planner, self.config.sim.vehicle_params) for _ in
                           range(self.n_opponents)]
        return out


class BasePlannerEnvWrapper(GymActionObservationWrapper, ABC):
    """A wrapper that adds a planner to the environment.

    Attributes:
        planner (PurePursuitPlanner): the planner
        planner_work (dict): the planner's work
        action_scaling (np.ndarray): the action scaling
        n_next_points (int): the number of next points to return
        skip_next_points (int): the number of next points to skip
        observation_space (gym.spaces.Box): the observation space
        action_applied_space (gym.spaces.Box): the action applied space
        # planner_action (nd.array): the planner's action
    """

    def __init__(self, env, config):
        super().__init__(env)
        self.config = config

        # self.n_next_points = config.planner.ego.n_next_points
        # self.skip_next_points = config.planner.ego.skip_next_points
        self.wpts_type = config.planner.ego.params.wpts_type

        if self.wpts_type == 'race':
            self.track = deepcopy(self.unwrapped.racing_line)
        elif self.wpts_type == 'center':
            self.track = deepcopy(self.unwrapped.center_line)
        self.waypoints = self.track.waypoints

        # Note: Important to deepcopy, otherwise the super().observation_space will be altered
        self.observation_space = deepcopy(self.get_wrapper_attr('observation_space'))

        config_planner = deepcopy(self.config.planner.ego.params)
        config_planner['vgain'] = float(np.random.choice(config_planner['vgain']))

        self.planner = make_planner(config_planner, self.config.sim.vehicle_params, self.track)

        self.observation_space['action_planner'] = gym.spaces.Box(
            low=-1,
            high=1,
            shape=(2,),
            dtype=np.float64
        )

        self.action_space_high = self.env.action_space.high
        self.action_space_low = self.env.action_space.low

        self.action_space = gym.spaces.Box(
            low=-1,
            high=1,
            shape=(2,),
            dtype=np.float64
        )

    def observation(self, obs):
        """Returns a modified observation."""
        # x, y, theta,
        steering, velocity = self.planner.plan(obs, self.waypoints[:, [0, 1, 2, 4]])

        # Rescale action from [-action_space_low, action_space_high] to [-1, 1]
        action_planner = \
            (np.array([steering, velocity]) - self.action_space_low) / (
                    self.action_space_high - self.action_space_low) * 2 - 1

        # Make sure that planner cannot generate actions that are not in the env's action space
        obs['action_planner'] = np.clip(
            action_planner,
            a_min=self.action_space.low,
            a_max=self.action_space.high
        )

        return obs

    def action(self, action):
        """Returns a modified action before :meth:`env.step` is called.

        Args:
            action (np.ndarray): the action that comes from the residual

        Returns:
            np.ndarray: the modified action that is combined with the planner's action
        """
        # Rescale action from [-1, 1] to [action_space_low, action_space_high]
        out = (self.action_space_high - self.action_space_low) * (action + 1) / 2 + self.action_space_low

        # use vectorized version with np where
        if self.config.safety_controller:
            safe_velocity = np.sqrt(0.8 * 9.81 / (np.tan(np.abs(out[0] + 1e-5)) / 0.33))
            out[1] = np.where(np.logical_and(abs(out[0]) > 0.2, safe_velocity < out[1]), safe_velocity, out[1])
        return out

    def step(self, *args, **kwargs):
        """Calls :meth:`env.step` and renders the environment."""
        out = super().step(*args, **kwargs)
        self.render()
        return out

    def reset(self, *args, **kwargs):
        """Calls :meth:`env.reset` and renders the environment."""
        out = super().reset(*args, **kwargs)
        self.render()
        config_planner = deepcopy(self.config.planner.ego.params)
        config_planner['vgain'] = float(np.random.choice(config_planner['vgain']))
        self.planner = make_planner(config_planner, self.config.sim.vehicle_params, self.track)
        return out


def make_planner(config_planner, vehicle_params, track=None):
    config_planner = deepcopy(config_planner)
    if config_planner.type == 'pure_pursuit':
        planner = PurePursuitPlanner(config_planner, vehicle_params)
    elif config_planner.type == 'apf':
        planner = PotentialFieldsPlanner(config_planner, vehicle_params)
    elif config_planner.type == 'ftg':
        planner = FTGPlanner(config_planner, vehicle_params)
    elif config_planner.type == 'disparity':
        planner = DisparityExtenderPlanner(config_planner, vehicle_params)
    elif config_planner.type == 'ftg_plus':
        planner = FTGPlusPlanner(config_planner, vehicle_params)
    elif config_planner.type == 'dummy':
        planner = DummyPlanner(config_planner)
    else:
        raise ValueError(f'Unknown planner type {config_planner.type}.')
    return planner

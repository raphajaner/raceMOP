import logging
from copy import deepcopy
import gymnasium as gym
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

from f110_gym.envs.base_classes import Simulator, Integrator
from f110_gym.envs.trackline import RacingLine, CenterLine, CircularQueue
from f110_gym.envs.env_utils import nearest_point_on_trajectory

WINDOW_W = 800
WINDOW_H = 800


class F110Env(gym.Env):
    """OpenAI gym environment for F1TENTH. Should be initialized by calling gym.make('f110_gym:f110-v1', **kwargs)."""

    metadata = {'render_modes': ['human', 'human_fast'], 'render_fps': 100}

    def __init__(self, config, map_names, seed=42, render_mode=None):
        self.config = config

        # Vehicle settings
        self.params = self.config.sim.vehicle_params
        self.num_agents = self.config.sim.n_agents
        self.timestep = self.config.sim.dt
        self.ego_idx = 0

        # Map and center lines
        self.map_name = map_names[0]
        self.map_path = self.config.maps.map_path + f'{self.map_name}/{self.map_name}_map'
        self.map_ext = self.config.maps.map_ext
        self.center_line = CenterLine(self.map_name, self.map_path, config_map=self.config.maps)
        self.racing_line = RacingLine(self.map_name, self.map_path, config_map=self.config.maps)
        self.start_position_line = self.racing_line if self.config.sim.start_positions.line_type else self.center_line

        self.start_pose_mode = self.config.sim.start_pose_mode
        self.start_pose_idx = 0
        self.start_positions_n_clusters = self.config.sim.start_positions.n_clusters
        self.start_pose = self._get_start_positions()
        self.n_positions = self.start_pose.shape[0]
        self.n_split = self.config.sim.start_positions.n_split
        if self.n_split is None:
            self.n_split = self.n_positions // self.num_agents

        # Rendering
        self.renderer = None
        self._render_mode = None
        self.render_mode = render_mode
        self.render_callbacks = []

        # Simulator
        # create seed from the current time
        self.seed = seed
        self.np_random = np.random.default_rng(seed=self.seed)
        self.sim = Simulator(self.params, self.num_agents, self.seed, self.timestep, Integrator.RK4)
        self.sim.set_map(self.map_path + '.yaml', self.map_ext)
        self.n_laps = 2

        # Reward params
        self.reward_params = self.config.sim.reward

        # States
        self._current_obs = None
        self.prev_progress = None
        self.near_starts = None
        self.toggle_list = None
        self.start_xs = None
        self.start_ys = None
        self.start_thetas = None
        self.start_rot = None

        # Others
        self.control_to_sim_ratio = int(self.config.sim['controller_dt'] / self.config.sim['dt'])

        # Race info
        self.lap_times = None
        self.lap_counts = None
        self.current_time = None
        self.progress_queue = None

        # Action space
        self.action_space = gym.spaces.Box(
            low=np.array([self.params['s_min'], self.params['v_min']]),
            high=np.array([self.params['s_max'], self.params['v_max']]),
            shape=(2,),
            dtype=np.float64
        )

        # Observation space
        self.observation_space = gym.spaces.Dict(
            {
                'aaa_scans': gym.spaces.Box(low=0, high=31, shape=(self.num_agents, 1080), dtype=np.float32),
                'poses_x': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_agents,), dtype=np.float64),
                'poses_y': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_agents,), dtype=np.float64),
                'poses_theta': gym.spaces.Box(low=-2 * np.pi, high=2 * np.pi, shape=(self.num_agents,),
                                              dtype=np.float64),
                'linear_vels_x': gym.spaces.Box(
                    low=self.params['v_min'], high=self.params['v_max'], shape=(self.num_agents,), dtype=np.float64
                ),
                'linear_vels_y': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_agents,), dtype=np.float64),
                'linear_vels': gym.spaces.Box(
                    low=self.params['v_min'], high=self.params['v_max'], shape=(self.num_agents,), dtype=np.float64
                ),
                'ang_vels_z': gym.spaces.Box(low=-100, high=100, shape=(self.num_agents,), dtype=np.float64),
                'collisions': gym.spaces.Box(low=0, high=1, shape=(self.num_agents,), dtype=bool),
                'collisions_env': gym.spaces.Box(low=0, high=1, shape=(self.num_agents,), dtype=bool),
                'collisions_overtaking_agents': gym.spaces.Box(low=0, high=1, shape=(self.num_agents,), dtype=bool),
                'collisions_overtaking_env': gym.spaces.Box(low=0, high=1, shape=(self.num_agents,), dtype=bool),
                'overtaking_success': gym.spaces.Box(low=0, high=1, shape=(1,), dtype=bool),
                'lap_times': gym.spaces.Box(low=0, high=np.inf, shape=(self.num_agents,), dtype=np.float64),
                'lap_counts': gym.spaces.Box(low=0, high=np.inf, shape=(self.num_agents,), dtype=np.int8),
                'prev_action': gym.spaces.Box(
                    low=np.repeat([[self.params['s_min'], self.params['v_min']]], self.num_agents, axis=0),
                    high=np.repeat([[self.params['s_max'], self.params['v_max']]], self.num_agents, axis=0),
                    shape=(self.num_agents, 2),
                    dtype=np.float64
                ),
                'slip_angle': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_agents,), dtype=np.float64),
                'yaw_rate': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_agents,), dtype=np.float64),
                'yaw_angle': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_agents,), dtype=np.float64),
                'acc_x': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_agents,), dtype=np.float64),
                'acc_y': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_agents,), dtype=np.float64),
                'steering_angle': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_agents,), dtype=np.float64),
                'progress': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float64),
                'safe_velocity': gym.spaces.Box(low=0.0, high=self.params['v_max'], shape=(1,), dtype=np.float64),
                'close_opponent': gym.spaces.Box(low=-20.0, high=20.0, shape=(1, 3), dtype=np.float64),
            }
        )

    @property
    def render_mode(self):
        """Get the rendering mode."""
        return self._render_mode

    @render_mode.setter
    def render_mode(self, render_mode):
        """Set the rendering mode."""
        if render_mode in ['human', 'human_fast']:
            self._render_mode = render_mode
            if self.renderer is None:
                # Import here to avoid import on non-gui systems
                from f110_gym.envs.rendering import EnvRenderer
                from pyglet import options as pygl_options
                pygl_options['debug_gl'] = False
                self.renderer = EnvRenderer(WINDOW_W, WINDOW_H)
                self.renderer.update_map(self.map_path, self.map_ext, self.map_name)
        else:
            if self.renderer is not None:
                self.renderer.close()
                self.renderer = None

    def step(self, action):
        """ Step function for the gym env."""

        if len(action.shape) < 2:
            action = np.expand_dims(action, 0)

        prev_obs = deepcopy(self._current_obs)  # Important to deepcopy here

        overtaking_success = False

        for _ in range(self.control_to_sim_ratio):
            obs = self.sim.step(action)
            self.current_time = self.current_time + self.timestep
            lap_done = self._check_lap(obs)

            # Reset opponent if it collides with the environment
            for idx in np.where(obs['collisions'][1:])[0] + 1:
                if obs['collisions'][idx] and not obs['collisions'][self.ego_idx]:
                    _, _, _, min_dist_segment = nearest_point_on_trajectory(self.sim.agent_poses[idx][:2],
                                                                            self.center_line.waypoints[:, :2])
                    self.sim.agents[idx].reset(self.center_line.waypoints[min_dist_segment])
                    logging.warning(f'Agent {idx} collided with the environment')

            if obs['overtaking_success'][0]:
                overtaking_success = True

            if obs['collisions'][self.ego_idx] or lap_done[self.ego_idx]:
                break

        obs['lap_times'] = self.lap_times
        obs['lap_counts'] = self.lap_counts
        obs['prev_action'] = action

        # Lap progress
        current_progress = self.center_line.calculate_progress(np.array([obs['poses_x'][0], obs['poses_y'][0]]))
        progress = np.clip(current_progress - self.prev_progress, 0.0, 0.2)
        if np.isnan(progress):
            progress = 0.0
            logging.warning('progress is nan')
        obs['progress'] = np.array([progress], dtype=np.float64)
        self.prev_progress = deepcopy(current_progress)

        # Check that ego vehicle did make progress over the last 10 steps or count as a collision
        self.progress_queue.add(progress)
        if self.progress_queue.max() <= 0.0:
            obs['collisions'][self.ego_idx] = True
            if self.sim.overtaking_idx[self.sim.close_opponent_idx]:
                obs['collisions_overtaking_env'][self.ego_idx] = True
            else:
                obs['collisions_env'][self.ego_idx] = True

        # Check if the episode is done when ego vehicle has completed 2 laps or has collided
        truncated = lap_done[self.ego_idx]
        terminated = obs['collisions'][self.ego_idx]

        # Check if overtaking was successful
        if overtaking_success:
            obs['overtaking_success'][0] = True

            # Dynamic reset of the overtaken agent for efficient training
            if self.config.sim.start_positions.dynamic_reset:
                self._dynamic_opponent_reset(obs)

        # Reward
        reward, reward_info = self._calc_reward(obs, prev_obs)
        info = reward_info

        self._current_obs = deepcopy(obs)

        if self.render_mode == "human" or self.render_mode == "human_fast":
            self.render()

        return obs, reward, terminated, truncated, deepcopy(info)

    def reset(self, options=None, seed=None, map_name=None):
        """Reset the environment."""

        # Reset the start positions
        if self.start_pose_mode == 'first':
            pass
        elif self.start_pose_mode == 'sequential':
            self.start_pose_idx += 1
        elif self.start_pose_mode == 'random':
            self.start_pose_idx = self.np_random.integers(0, self.n_positions)
        else:
            raise ValueError(f'Unknown start_pose_mode: {self.start_pose_mode}')
        idx_range = self.start_pose_idx + np.arange(0, self.num_agents * self.n_split, self.n_split)
        if self.config.sim.start_positions.dynamic_reset and len(idx_range) > 2:
            idx_range[-1] = self.start_pose_idx - self.n_split
        idx_range %= self.n_positions

        poses = self.start_pose[idx_range]

        # Initialize attributes
        self.current_time = 0.0
        self.near_starts = np.array([True] * self.num_agents)
        self.toggle_list = np.zeros((self.num_agents,))

        # states after reset
        self.start_xs = poses[:, 0]
        self.start_ys = poses[:, 1]
        self.start_thetas = poses[:, 2]
        self.start_rot = np.array(
            [[np.cos(-self.start_thetas[0]), -np.sin(-self.start_thetas[0])],
             [np.sin(-self.start_thetas[0]), np.cos(-self.start_thetas[0])]]
        )

        self.lap_times = np.zeros((self.num_agents,))
        self.lap_counts = np.zeros((self.num_agents,), dtype=np.int8)

        # Reset the simulator and correct observations
        obs = self.sim.reset(poses)

        obs['lap_times'] = self.lap_times
        obs['lap_counts'] = self.lap_counts
        obs['prev_action'] = np.zeros((self.num_agents, 2))
        obs['progress'] = np.array([0.0], dtype=np.float64)

        self.prev_progress = self.center_line.calculate_progress(np.array([obs['poses_x'][0], obs['poses_y'][0]]))
        self.progress_queue = CircularQueue(100)

        self._current_obs = deepcopy(obs)
        _, reward_info = self._calc_reward(obs, obs)
        # Set all entries in the reward_info to 0
        info = {key: 0.0 for key in reward_info.keys()}

        if self.render_mode == "human" or self.render_mode == "human_fast":
            self.render()

        return obs, info

    def render(self):
        """Render the environment."""
        if self.renderer is not None:
            self.renderer.update_obs(self._current_obs)
            for render_callback in self.render_callbacks:
                render_callback(self.renderer)
            self.renderer.dispatch_events()
            self.renderer.on_draw()
            self.renderer.flip()
            if self.render_mode == 'human':
                time.sleep(0.005)
            elif self.render_mode == 'human_fast':
                pass

    def close(self):
        """Close the environment."""
        if self.renderer is not None:
            self.renderer.close()
        self.renderer = None
        super().close()

    def _calc_reward(self, obs, prev_obs):

        reward_info = dict()

        # Taking a step
        if self.reward_params.step != 0.0:
            reward_info['reward_step'] = 1 * self.reward_params.step

        # Distance to the closest obstacle
        if self.reward_params.distance_threshold != 0.0:
            distance = np.min(np.abs(obs['aaa_scans'][0]))
            if distance < self.reward_params.distance_threshold:
                reward_info['reward_distance'] = (self.reward_params.distance_threshold - distance) \
                                                 * self.reward_params.safe_distance_to_obstacles
            else:
                reward_info['reward_distance'] = 0.0

        # Lap progress
        if self.reward_params.progress != 0.0:
            reward_info['reward_progress'] = obs['progress'][0] * self.reward_params.progress

        # Traveled distance
        if self.reward_params.traveled_distance != 0.0:
            reward_info['reward_traveled_distance'] = obs['linear_vels_x'][0] * (
                    self.timestep * self.control_to_sim_ratio) * self.reward_params.traveled_distance

        # Safe velocity
        if self.reward_params.safe_velocity != 0.0:
            reward_info['reward_safe_velocity'] = \
                (2.0 - np.abs(obs['linear_vels_x'][0] - obs['safe_velocity'][0])) * self.reward_params.safe_velocity

        # Collision
        if self.reward_params.collision != 0.0:
            reward_info['reward_collision'] = obs['collisions'][0] * self.reward_params.collision

        # Longitudinal velocity
        if self.reward_params.long_vel != 0.0:
            reward_info['reward_long_vel'] = obs['linear_vels_x'][0] * self.reward_params.long_vel

        # Lateral velocity
        if self.reward_params.lat_vel != 0.0:
            reward_info['reward_lat_vel'] = np.abs(obs['linear_vels_y'][0]) * self.reward_params.lat_vel

        # Action change
        if self.reward_params.action_change != 0.0:
            reward_info['reward_action_change'] = np.sum(np.abs(obs['prev_action'][0] - prev_obs['prev_action'][0]) / (
                    self.action_space.high - self.action_space.low)) * self.reward_params.action_change

        # Yaw rate change
        if self.reward_params.yaw_change != 0.0:
            reward_info['reward_yaw_change'] = np.abs(
                obs['yaw_rate'][0] - prev_obs['yaw_rate'][0]) * self.reward_params.yaw_change

        # Overtaking
        if self.reward_params.overtaking != 0.0:
            reward_info['reward_overtaking'] = self.reward_params.overtaking if obs['overtaking_success'][0] else 0.0

        # Total reward
        reward = np.sum(list(reward_info.values())) * self.reward_params.scaling
        reward_info['reward'] = reward

        return reward, reward_info

    def _get_start_positions(self):
        """Get the start positions for the vehicles."""

        select_idx = np.abs(self.start_position_line.waypoints[:, 3]) < self.config.sim.start_positions.kappa_threshold

        waypoints = self.start_position_line.waypoints[select_idx]
        ss = self.start_position_line.ss[select_idx]

        # Find the start positions using kmeans
        cluster_points = np.stack((waypoints[:, 0], waypoints[:, 1], ss), axis=1)
        kmeans = KMeans(n_clusters=self.start_positions_n_clusters, random_state=0, n_init='auto').fit(
            cluster_points)
        # Get one waypoint from each cluster that is closest to the cluster center
        start_poses = np.zeros((self.start_positions_n_clusters, 5))
        track_ss = np.zeros((self.start_positions_n_clusters,))

        for i in range(self.start_positions_n_clusters):
            dists = np.linalg.norm(waypoints[kmeans.labels_ == i, :2] - kmeans.cluster_centers_[i, :2], axis=1)
            idx = np.argmin(dists)
            track_ss[i] = ss[kmeans.labels_ == i][idx]
            start_poses[i] = waypoints[kmeans.labels_ == i, :][idx]

        # reorder the poses again so that they are in order of the track
        start_poses = start_poses[track_ss.argsort()]

        do_plot = False
        if do_plot:
            plt.figure()
            plt.plot(self.start_position_line.waypoints[:, 0], self.start_position_line.waypoints[:, 1],
                     color='black', linewidth=1)
            for i, start_pose in enumerate(start_poses):
                cluster = waypoints[kmeans.labels_ == i, :2]
                c = np.random.rand(3, )
                plt.scatter(cluster[:, 0], cluster[:, 1], c=c, s=5)
                plt.scatter(start_pose[0], start_pose[1], c=c, marker='o', s=30)
                plt.scatter(kmeans.cluster_centers_[i][0], kmeans.cluster_centers_[i][1], c='black', marker='*', s=30)
                plt.annotate(str(i), (start_pose[0], start_pose[1]))
            plt.show()
            plt.close()

        return start_poses

    def _dynamic_opponent_reset(self, obs):
        """Reset the opponent vehicle to a new position after being overtaken by the ego vehicle."""
        ego_pose = np.array([obs['poses_x'][0], obs['poses_y'][0]])
        # find the position on the self.start_position_line that is 5m ahead of the ego vehicle
        _, _, _, closest_idx = nearest_point_on_trajectory(ego_pose, self.start_position_line.waypoints[:, :2])

        reset_dist = 15.0
        next_idx = closest_idx
        while True:
            next_idx += 1
            if next_idx >= len(self.start_position_line.waypoints):
                next_idx = 0
            if (self.start_position_line.ss[next_idx] > self.start_position_line.ss[closest_idx] and
                self.start_position_line.ss[next_idx] - self.start_position_line.ss[closest_idx] > reset_dist) or \
                    (self.start_position_line.ss[next_idx] < self.start_position_line.ss[closest_idx] and
                     self.start_position_line.ss[next_idx] + self.start_position_line.ss[-1] -
                     self.start_position_line.ss[closest_idx] > reset_dist):
                break

        pose_idx = np.argmin(np.linalg.norm(ego_pose - self.start_pose[:, :2], axis=1))
        n_split = self.config.sim.start_positions.n_split
        n_split = n_split if n_split is not None else len(self.start_pose) // self.num_agents
        pose_idx = (pose_idx + n_split) % len(self.start_pose)
        # Reset the agent further away from the ego vehicle
        agent_dists = np.linalg.norm(ego_pose - self.sim.agent_poses[1:, :2], axis=1)
        agent_idx = np.argmax(agent_dists)
        # import pdb
        # pdb.set_trace()
        self.sim.agents[agent_idx + 1].reset(self.start_position_line.waypoints[next_idx][:3])
        # self.sim.agents[agent_idx + 1].reset(self.start_pose[pose_idx])
        # distance from this to ego
        # distance = np.linalg.norm(self.start_position_line.waypoints[next_idx][:2] - ego_pose)

    def _check_lap(self, obs):
        """ Check if the current rollout is done."""
        # This is assuming 2 agents.py
        left_t = 2
        right_t = 2

        poses_x = np.array(obs['poses_x']) - self.start_xs
        poses_y = np.array(obs['poses_y']) - self.start_ys
        delta_pt = np.dot(self.start_rot, np.stack((poses_x, poses_y), axis=0))
        temp_y = delta_pt[1, :]
        idx1 = temp_y > left_t
        idx2 = temp_y < -right_t
        temp_y[idx1] -= left_t
        temp_y[idx2] = -right_t - temp_y[idx2]
        temp_y[np.invert(np.logical_or(idx1, idx2))] = 0

        dist2 = delta_pt[0, :] ** 2 + temp_y ** 2
        closes = dist2 <= 0.1
        for i in range(self.num_agents):
            if closes[i] and not self.near_starts[i]:
                self.near_starts[i] = True
                self.toggle_list[i] += 1
            elif not closes[i] and self.near_starts[i]:
                self.near_starts[i] = False
                self.toggle_list[i] += 1
            self.lap_counts[i] = self.toggle_list[i] // 2
            if self.toggle_list[i] <= 2 * self.n_laps:
                self.lap_times[i] = self.current_time

        # Only look at the ego vehicle
        lap_done = (self.toggle_list >= 2 * self.n_laps)
        return lap_done

    def update_params(self, params, index=-1):
        """Update the vehicle parameters."""
        self.sim.update_params(params, agent_idx=index)

    def add_render_callback(self, callback_func):
        """Add extra drawing function to call during rendering."""
        self.render_callbacks.append(callback_func)

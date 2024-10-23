import warnings
from enum import Enum

import numpy as np

from f110_gym.envs.collision_models import collision_multiple
from f110_gym.envs.dynamic_models import vehicle_dynamics_st, pid
from f110_gym.envs.env_utils import get_vertices
from f110_gym.envs.env_utils import pi_2_pi
from f110_gym.envs.laser_models import ScanSimulator2D, check_ttc_jit, ray_cast


class Integrator(Enum):
    RK4 = 1
    Euler = 2


class RaceCar(object):
    """ Base level race car class, handles the physics and laser scan of a single vehicle."""

    def __init__(self, params, seed, is_ego=False, time_step=0.01, num_beams=1080, fov=4.7,
                 integrator=Integrator.Euler):
        """Init function."""

        self.scan_simulator = None
        self.cosines = None
        self.scan_angles = None
        self.side_distances = None

        self.params = params
        # Accessing params from dict is expensive, so we cache them here
        self.mu = params['mu']
        self.C_Sf = params['C_Sf']
        self.C_Sr = params['C_Sr']
        self.lf = params['lf']
        self.lr = params['lr']
        self.h = params['h']
        self.m = params['m']
        self.I = params['I']
        self.s_min = params['s_min']
        self.s_max = params['s_max']
        self.sv_min = params['sv_min']
        self.sv_max = params['sv_max']
        self.v_switch = params['v_switch']
        self.a_max = params['a_max']
        self.v_min = params['v_min']
        self.v_max = params['v_max']
        self.length = params['length']
        self.width = params['width']
        self.wheelbase = params['lf'] + params['lr']
        self.lidar_noise = 0.01

        self.seed = seed
        self.is_ego = is_ego
        self.time_step = time_step
        self.num_beams = num_beams
        self.fov = fov
        self.integrator = integrator
        # state is [x, y, steer_angle, vel, yaw_angle, yaw_rate, slip_angle]
        self.state = np.zeros((7,))
        # Add another state: Lateral acceleration (no independent state)
        self.a_x = np.zeros((1,))
        self.a_y = np.zeros((1,))

        self.max_slip = None
        self.max_yaw = None

        # pose of opponents in the world
        self.opp_poses = None

        # control inputs
        self.accel = 0.0
        self.steer_angle_vel = 0.0

        # steering delay buffer
        self.steer_buffer = np.empty((0,))
        self.steer_buffer_size = 2

        # collision identifier
        self.in_collision = False

        # collision threshold for iTTC to environment
        self.ttc_thresh = 0.005

        # initialize scan sim
        if self.scan_simulator is None:
            self.scan_rng = np.random.default_rng(seed=self.seed)
            self.scan_simulator = ScanSimulator2D(num_beams, fov)

            scan_ang_incr = self.scan_simulator.get_increment()

            # angles of each scan beam, distance from lidar to edge of car at each beam,
            # and precomputed cosines of each angle
            self.cosines = np.zeros((num_beams,))
            self.scan_angles = np.zeros((num_beams,))
            self.side_distances = np.zeros((num_beams,))

            dist_sides = self.width / 2.
            dist_fr = (self.lf + self.lr) / 2.

            for i in range(num_beams):
                angle = -fov / 2. + i * scan_ang_incr
                self.scan_angles[i] = angle
                self.cosines[i] = np.cos(angle)

                if angle > 0:
                    if angle < np.pi / 2:
                        # between 0 and pi/2
                        to_side = dist_sides / np.sin(angle)
                        to_fr = dist_fr / np.cos(angle)
                        self.side_distances[i] = min(to_side, to_fr)
                    else:
                        # between pi/2 and pi
                        to_side = dist_sides / np.cos(angle - np.pi / 2.)
                        to_fr = dist_fr / np.sin(angle - np.pi / 2.)
                        self.side_distances[i] = min(to_side, to_fr)
                else:
                    if angle > -np.pi / 2:
                        # between 0 and -pi/2
                        to_side = dist_sides / np.sin(-angle)
                        to_fr = dist_fr / np.cos(-angle)
                        self.side_distances[i] = min(to_side, to_fr)
                    else:
                        # between -pi/2 and -pi
                        to_side = dist_sides / np.cos(-angle - np.pi / 2)
                        to_fr = dist_fr / np.sin(-angle - np.pi / 2)
                        self.side_distances[i] = min(to_side, to_fr)

    def update_params(self, params):
        """ Updates the physical parameters of the vehicle with params."""
        self.params = params
        self.mu = params['mu']
        self.C_Sf = params['C_Sf']
        self.C_Sr = params['C_Sr']
        self.lf = params['lf']
        self.lr = params['lr']
        self.h = params['h']
        self.m = params['m']
        self.I = params['I']
        self.s_min = params['s_min']
        self.s_max = params['s_max']
        self.sv_min = params['sv_min']
        self.sv_max = params['sv_max']
        self.v_switch = params['v_switch']
        self.a_max = params['a_max']
        self.v_min = params['v_min']
        self.v_max = params['v_max']
        self.length = params['length']
        self.width = params['width']
        self.wheelbase = params['lf'] + params['lr']

    def set_map(self, map_path, map_ext):
        """ Sets the map for scan simulator."""
        self.scan_simulator.set_map(map_path, map_ext)

    def reset(self, pose):
        """ Resets the vehicle to a pose.

        Args:
            pose (np.ndarray (3, )): pose to reset the vehicle to
        """
        # clear control inputs
        self.accel = 0.0
        self.steer_angle_vel = 0.0
        # clear state
        self.state = np.zeros((7,))
        self.state[0:2] = pose[0:2]
        self.state[4] = pose[2]
        self.steer_buffer = np.empty((0,))

        self.a_x = np.zeros((1,))
        self.a_y = np.zeros((1,))

        self.in_collision = False

        self.scan_rng = np.random.default_rng(seed=self.seed)

    def ray_cast_agents(self, scan):
        """ Ray cast onto other agents.py in the env, modify original scan."""
        new_scan = scan
        # loop over all opponent vehicle poses
        for opp_pose in self.opp_poses:
            # get vertices of current opponent
            opp_vertices = get_vertices(opp_pose, self.length, self.width)
            new_scan = ray_cast(np.append(self.state[0:2], self.state[4]), new_scan, self.scan_angles, opp_vertices)
        return new_scan

    def check_ttc(self, current_scan):
        """ Check iTTC against the environment.

        Sets vehicle states accordingly if collision occurs. Note that this does NOT check collision with other
        agents.py. State is [x, y, steer_angle, vel, yaw_angle, yaw_rate, slip_angle]
        Args:
            current_scan(np.ndarray, (n, )): current scan range array
        """

        in_collision = check_ttc_jit(current_scan, self.state[3], self.scan_angles, self.cosines, self.side_distances,
                                     self.ttc_thresh)

        # if in collision stop vehicle
        if in_collision:
            self.state[3:] = 0.
            self.accel = 0.0
            self.steer_angle_vel = 0.0

        # update state
        self.in_collision = in_collision

        return in_collision

    def update_pose(self, raw_steer, vel):
        """ Steps the vehicle's physical simulation.

        The State is [x, y, steer_angle, vel, yaw_angle, yaw_rate, slip_angle]

        Args:
            steer (float): desired steering angle
            vel (float): desired longitudinal velocity

        Returns:
            current_scan
        """

        # steering delay
        if self.steer_buffer.shape[0] < self.steer_buffer_size:
            steer = 0.0
            self.steer_buffer = np.append(raw_steer, self.steer_buffer)
        else:
            steer = self.steer_buffer[-1]
            self.steer_buffer = self.steer_buffer[:-1]
            self.steer_buffer = np.append(raw_steer, self.steer_buffer)

        # steering angle velocity input to steering velocity acceleration input
        accl, sv = pid(vel, steer, self.state[3], self.state[2], self.sv_max, self.a_max,
                       self.v_max, self.v_min)

        # Note: accl and sv get saturated in the integrator
        if self.integrator is Integrator.RK4:
            # RK4 integration
            k1 = self._get_vehicle_dynamics_st(self.state, sv, accl)

            k2_state = self.state + self.time_step * (k1 / 2)
            k2 = self._get_vehicle_dynamics_st(k2_state, sv, accl)

            k3_state = self.state + self.time_step * (k2 / 2)
            k3 = self._get_vehicle_dynamics_st(k3_state, sv, accl)

            k4_state = self.state + self.time_step * k3
            k4 = self._get_vehicle_dynamics_st(k4_state, sv, accl)

            # dynamics integration
            f = (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
            self.a_x = f[3] * np.cos(self.state[6])
            self.a_y = f[3] * np.sin(self.state[6])
            self.state = self.state + self.time_step * f

        elif self.integrator is Integrator.Euler:
            f = self._get_vehicle_dynamics_st(self.state, sv, accl)
            self.a_x = f[3] * np.cos(self.state[6])
            self.a_y = f[3] * np.sin(self.state[6])
            self.state = self.state + self.time_step * f

        # bound yaw angle
        if self.state[4] > 2 * np.pi:
            self.state[4] = self.state[4] - 2 * np.pi
        elif self.state[4] < 0:
            self.state[4] = self.state[4] + 2 * np.pi

        # update scan
        current_scan = self.scan_simulator.scan(np.append(self.state[0:2], self.state[4]), self.scan_rng,
                                                std_dev=self.lidar_noise)

        return current_scan

    def _get_vehicle_dynamics_st(self, state, sv, accl):
        f = vehicle_dynamics_st(
            state,
            np.array([sv, accl]),
            self.mu,
            self.C_Sf,
            self.C_Sr,
            self.lf,
            self.lr,
            self.h,
            self.m,
            self.I,
            self.s_min,
            self.s_max,
            self.sv_min,
            self.sv_max,
            self.v_switch,
            self.a_max,
            self.v_min,
            self.v_max
        )
        return f

    def update_opp_poses(self, opp_poses):
        """ Updates the vehicle's information on other vehicles.

        Args:
            opp_poses (np.ndarray(num_other_agents, 3)): updated poses of other agents.py
        """
        self.opp_poses = opp_poses

    def update_scan(self, agent_scans, agent_index):
        """ Steps the vehicle's laser scan simulation.

        Separated from update_pose because needs to update scan based on NEW poses of agents.py in the environment

        Args:
            agents.py scans list (modified in-place),
            agents.py index (int)
        """

        current_scan = agent_scans[agent_index]
        # check ttc
        self.check_ttc(current_scan)
        # ray cast other agents.py to modify scan
        new_scan = self.ray_cast_agents(current_scan)
        agent_scans[agent_index] = new_scan


class Simulator(object):
    """Simulator class, handles the interaction and update of all vehicles in the environment.

    Attributes:
        num_agents (int): number of agents.py in the environment
        time_step (float): physics time step
        agent_poses (np.ndarray(num_agents, 3)): all poses of all agents.py
        agents (list[RaceCar]): container for RaceCar objects
        collisions (np.ndarray(num_agents, )): array of collision indicator for each agents.py
        collision_idx (np.ndarray(num_agents, )): which agents.py is each agents.py in collision with

    """

    def __init__(self, params, num_agents, seed, time_step=0.01, ego_idx=0, integrator=Integrator.RK4):
        """Init function.

        Args:
            params (dict): dictionary of parameters for the environment, see details below
            num_agents (int): number of agents.py in the environment
            seed (int): random seed for the environment
            time_step (float, default=0.01): physics time step
            ego_idx (int, default=0): index of the ego vehicle
            integrator (Integrator, default=Integrator.RK4): integrator for the vehicle dynamics
        """

        self.num_agents = num_agents
        self.seed = seed
        self.time_step = time_step
        self.ego_idx = ego_idx
        self.params = params
        self.length = params['length']
        self.width = params['width']
        self.agent_poses = np.empty((self.num_agents, 3))
        self.agents = []

        # To track collisions and overtaking
        self.collisions = np.zeros((self.num_agents,))
        self.collision_idx = -1 * np.ones((self.num_agents,))
        self.overtaking_idx = np.zeros((self.num_agents,))
        self.close_opponent_idx = None

        # initializing agents.py
        for i in range(self.num_agents):
            agent = RaceCar(params, self.seed, is_ego=i == ego_idx, time_step=self.time_step, integrator=integrator)
            self.agents.append(agent)

    def set_map(self, map_path, map_ext):
        """Sets the map of the environment and sets the map for scan simulator of each agents.py.

        Args:
            map_path (str): path to the map yaml file
            map_ext (str): extension for the map image file
        """
        for agent in self.agents:
            agent.set_map(map_path, map_ext)

    def update_params(self, params, agent_idx=-1):
        """Updates the params of agents.py, if an index of an agents.py is given, update only that agents.py's params

        Args:
            params (dict): dictionary of params, see details in docstring of __init__
            agent_idx (int, default=-1): index for agents.py that needs param update, if negative, update all agents.py
        """
        if agent_idx < 0:
            # Update params for all
            for agent in self.agents:
                agent.update_params(params)
        elif 0 <= agent_idx < self.num_agents:
            # Update params for a specific agent
            self.agents[agent_idx].update_params(params)
        else:
            raise IndexError('Index given is out of bounds for list of agents.py.')

    def check_collision(self):
        """Checks for collision between agents.py using GJK and agents.py' body vertices."""
        all_vertices = np.empty((self.num_agents, 4, 2))  # get vertices of all agents.py
        for i in range(self.num_agents):
            all_vertices[i, :, :] = get_vertices(np.append(self.agents[i].state[0:2], self.agents[i].state[4]),
                                                 self.length, self.width)
        collisions, collision_idx = collision_multiple(all_vertices)
        return collisions, collision_idx

    def check_overtaking(self):
        states = np.array([agent.state[[0, 1, 4]] for agent in self.agents])

        success = self._compare_positions_with_direction(
            states[0, 0], states[0, 1], states[0, 2],  # Ego vehicle
            states[1:, 0], states[1:, 1], states[1:, 2]  # Other vehicles
        )
        return success

    def _compare_positions_with_direction(self, x_a, y_a, theta_a, x_b, y_b, theta_b):
        diff_x = np.array(x_b - x_a)
        diff_y = np.array(y_b - y_a)

        distances = np.sqrt(diff_x ** 2 + diff_y ** 2)
        indices = np.where(distances < 4.0)[0]

        # set all other overtaking indices to false
        n_indices = np.where(distances >= 7.0)[0]
        self.overtaking_idx[n_indices + 1] = False

        if indices.size == 0:
            return False

        for idx in indices:
            diff_y_ = [diff_y[idx]]
            diff_x_ = [diff_x[idx]]
            # Angle from vehicle A to vehicle B
            angle_to_point_b = np.arctan2(diff_y_, diff_x_)
            theta_a = pi_2_pi(theta_a)

            diff_angle = pi_2_pi(angle_to_point_b - theta_a)
            diff_angle = np.rad2deg(diff_angle)

            # get orientation angle between the cars, ie., if the cars are driving in the same direction
            # or if the cars are driving in opposite directions
            orientation_angle = pi_2_pi(theta_a - theta_b[idx])
            # orientation_angle = np.rad2deg(orientation_angle)

            success = False
            if -90.0 <= diff_angle <= 90.0:
                # print("Vehicle B is in front")
                # Overtaking has started
                self.overtaking_idx[idx + 1] = True
            else:
                # both cars must drive in the same direction
                if self.overtaking_idx[idx + 1]:
                    if abs(orientation_angle) < np.pi / 2:
                        # print("Vehicle A is in front")
                        # project distance in the direction of the vehicle B
                        distance = np.sqrt(diff_x[idx] ** 2 + diff_y[idx] ** 2)
                        projected_distance = abs(distance * np.cos(diff_angle))
                        # projected_distance = abs(distance * np.cos(theta_b[idx]))
                        # print(f"Projected distance: {projected_distance}")
                        if projected_distance > 1.2:
                            self.overtaking_idx[idx + 1] = False
                            success = True
                            # print(f'Successful overtaking of vehicle {idx + 1} at distance {projected_distance}')
                    else:
                        warnings.warn("Vehicles are driving in opposite directions. This should not happen.")

            return success

    def step(self, control_inputs):
        """Steps the simulation environment.

        Args:
            control_inputs (np.ndarray (num_agents, 2)): control inputs of all agents.py,
            first column is desired steering angle, second column is desired velocity
        
        Returns:
            observations (dict): dictionary for observations: poses of agents.py,
            current laser scan of each agents.py, collision indicators, etc.
        """

        agent_scans = []

        # looping over agents.py
        for i, agent in enumerate(self.agents):
            # update each agents.py's pose
            current_scan = agent.update_pose(control_inputs[i, 0], control_inputs[i, 1])
            agent_scans.append(current_scan)
            # update sim's information of agents.py poses
            self.agent_poses[i, :] = np.append(agent.state[0:2], agent.state[4])

        dist_close_all = []
        for i in range(1, self.num_agents):
            dist_close = np.linalg.norm(self.agent_poses[0, 0:2] - self.agent_poses[i, 0:2])
            dist_close_all.append(dist_close)
        close_opponent_idx = np.argmin(dist_close_all) + 1
        close_opponent = np.array([self.agent_poses[close_opponent_idx, :]])
        self.close_opponent_idx = close_opponent_idx

        # check collisions between all agents
        collisions_agents, collision_idx_agents = self.check_collision()

        self.collisions = collisions_agents.copy()
        self.collision_idx = collision_idx_agents

        # collision with environment
        collisions_env = np.zeros((self.num_agents,))
        collisions_overtaking_env = np.zeros((self.num_agents,))

        for i, agent in enumerate(self.agents):
            # update agents.py's information on other agents.py
            opp_poses = np.concatenate((self.agent_poses[0:i, :], self.agent_poses[i + 1:, :]), axis=0)
            agent.update_opp_poses(opp_poses)
            # update each agents.py's current scan based on other agents.py
            agent.update_scan(agent_scans, i)

            # update agents.py collision with environment, i.e., .in_collision is only for environment collisions
            if agent.in_collision:
                # if i == 0 and in overtaking
                if i == 0:
                    if self.overtaking_idx[close_opponent_idx]:
                        collisions_overtaking_env[i] = 1.0
                    else:
                        collisions_env[i] = 1.0
                self.collisions[i] = 1.0

        overtaking_success = self.check_overtaking()

        observations = dict()
        observations['aaa_scans'] = np.array([scan for scan in agent_scans]).astype(np.float32)

        observations['poses_x'] = np.array([a.state[0] for a in self.agents])
        observations['poses_y'] = np.array([a.state[1] for a in self.agents])

        observations['steering_angle'] = np.array([a.state[2] for a in self.agents])
        observations['linear_vels'] = np.array([a.state[3] for a in self.agents])
        observations['linear_vels_x'] = np.array([np.cos(a.state[6]) * a.state[3] for a in self.agents])
        observations['linear_vels_y'] = np.array(
            [np.sin(a.state[6]) * a.state[3] if a.state.shape[-1] == 7 else 0.0 for a in self.agents])

        observations['poses_theta'] = np.array([a.state[4] for a in self.agents])
        observations['yaw_rate'] = np.array([a.state[5] for a in self.agents])
        observations['ang_vels_z'] = np.array([a.state[5] for a in self.agents])
        observations['yaw_angle'] = np.array([a.state[4] for a in self.agents])
        observations['slip_angle'] = np.array([a.state[6] for a in self.agents])

        observations['acc_x'] = np.array([a.a_x for a in self.agents])
        observations['acc_y'] = np.array([a.a_y for a in self.agents])

        observations['collisions'] = np.array([bool(self.collisions[i]) for i in range(self.num_agents)])
        observations['collisions_env'] = np.array([bool(collisions_env[i]) for i in range(self.num_agents)])
        observations['collisions_overtaking_env'] = np.array(
            [bool(collisions_overtaking_env[i]) for i in range(self.num_agents)])
        observations['collisions_overtaking_agents'] = np.array(
            [bool(collisions_agents[i]) for i in range(self.num_agents)])

        assert observations['collisions'][0] == observations['collisions_env'][0] + \
               observations['collisions_overtaking_env'][0] + observations['collisions_overtaking_agents'][0]

        safe_velocity = np.sqrt(self.agents[0].mu * 9.81 / (
                np.tan(np.abs(observations['steering_angle'][0]) + 1e-5) / self.agents[0].wheelbase))
        observations['safe_velocity'] = np.clip(np.array([safe_velocity]), 0.0, self.agents[0].v_max)

        # These observations are only for the ego vehicle
        observations['overtaking_success'] = np.array([bool(overtaking_success)])
        observations['close_opponent'] = close_opponent if close_opponent is not None else np.zeros((1, 3))

        return observations

    def reset(self, poses):
        """Resets the simulation environment by given poses."""

        assert poses.shape[0] == self.num_agents, 'Number of poses for reset does not match number of agents.py.'

        # loop over poses to reset
        for i in range(self.num_agents):
            self.agents[i].reset(poses[i, :])

        # Dummy action to get the first observation
        obs = self.step(np.zeros((self.num_agents, 2)))
        return obs

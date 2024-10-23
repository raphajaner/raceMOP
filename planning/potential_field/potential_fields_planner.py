import time
from copy import deepcopy
from functools import partial

import numpy as np
from matplotlib import pyplot as plt
from numba import njit
from scipy.interpolate import *

from f110_gym.envs.env_utils import distance_between_points, find_close_obstacles
from f110_gym.envs.env_utils import nearest_point_on_trajectory, first_point_on_trajectory_intersecting_circle, \
    get_actuation
from planning.potential_field.potential_field import plot_potential_fields, step_potential_field


@njit(cache=True)
def find_target_point_all(scans):
    x_map_scans = scans[:, 0]
    y_map_scans = scans[:, 1]
    gaps = []
    angles = np.arctan2(y_map_scans, x_map_scans)

    for i in range(1, x_map_scans.shape[0] - 1):
        left = i - 1
        dis_1 = np.sqrt((x_map_scans[i] - x_map_scans[left]) ** 2 + (y_map_scans[i] - y_map_scans[left]) ** 2)
        if dis_1 > 1.0:
            angle = (angles[left] + angles[i]) / 2.0
            # get max length of the two points
            len_1 = np.sqrt(np.sum((np.array([x_map_scans[left], y_map_scans[left]])) ** 2))
            len_2 = np.sqrt(np.sum((np.array([x_map_scans[i], y_map_scans[i]])) ** 2))
            point_length = np.max(np.array([len_1, len_2]))
            point = np.array([np.cos(angle), np.sin(angle)]) * point_length
            if point[1] > 1.0:
                gaps.append(point)

    if len(gaps) == 0:
        x_map_scans = x_map_scans[y_map_scans > 0.0]
        y_map_scans = y_map_scans[y_map_scans > 0.0]

        # check that scans not empty otherwise print the line number
        if x_map_scans.shape[0] > 0:
            idx = np.argmax(np.sqrt(x_map_scans ** 2 + y_map_scans ** 2))
            gaps.append(np.array([x_map_scans[idx], y_map_scans[idx]]))
        else:
            print('scans empty')
            print(f'x_map_scans: {x_map_scans}, y_map_scans: {y_map_scans}')
    return gaps


@njit
def _postprocess_path(path):
    # Filter the path
    new_path = [path[0]]
    for point in path[1:]:
        if distance_between_points(new_path[-1], point) > 0.1:
            new_path.append(point)
        else:
            new_path[-1] = (new_path[-1] + point) / 2
    return new_path


def postprocess_path(path):
    path_smooth = _postprocess_path(path)
    return np.array(path_smooth)


@njit(cache=True)
def get_velocity_profile(goal, max_speed, steering):
    # 1) Velocity based on the distance to the target point
    target_dis = np.sqrt(np.sum(goal ** 2))  # , axis=1)).mean()
    if target_dis <= 2.0:
        velocity_dis = max_speed / 3.0
    elif target_dis <= 3.0:
        velocity_dis = max_speed / 2.5
    elif target_dis <= 6.0:
        velocity_dis = max_speed / 1.7
    elif target_dis <= 8.0:
        velocity_dis = max_speed / 1.2
    else:
        velocity_dis = max_speed

    # 2) Velocity based on physical constraints
    velocity_con = np.sqrt(0.8 * 9.81 / (np.tan(np.abs(steering) + 1e-5) / 0.33))

    # Take the minimum of the velocities
    return np.min(np.array([velocity_dis, velocity_con], dtype=np.float64))


@njit(cache=True)
def preprocess_scans(scans_xy, threshold_gap=0.1, threshold_min=-4.0, threshold_max=10.0):
    """ Filter the map_scans points that do not exceed the threshold distance from the others or are too far away. """
    # Remove all points that are behind the car
    scans_xy = scans_xy[scans_xy[:, 1] > threshold_min]

    filtered_scans_xy = np.zeros_like(scans_xy)
    filtered_scans_xy[0] = scans_xy[0]
    idx = 0
    for i in range(1, scans_xy.shape[0]):
        if (distance_between_points(filtered_scans_xy[idx], scans_xy[i]) > threshold_gap
                and distance_between_points(scans_xy[i], np.array([0.0, 0.0])) < threshold_max):
            idx += 1
            filtered_scans_xy[idx] = scans_xy[i]
    # Cut off the rest of the array
    filtered_scans_xy = filtered_scans_xy[:idx + 1]

    # Close the gap that is behind the car from x_map_scans[0] to x_map_scans[-1] by interpolating
    fill_gap_x = np.linspace(filtered_scans_xy[-1, 0], filtered_scans_xy[1, 0], 5)
    fill_gap_y = np.linspace(filtered_scans_xy[-1, 1], filtered_scans_xy[1, 1], 5)
    fill_gap_xy = np.vstack((fill_gap_x, fill_gap_y)).T
    # Stack to other points
    filtered_scans_xy = np.vstack((filtered_scans_xy, fill_gap_xy))
    return filtered_scans_xy


def plan_callback(obs, prev_goal, prev_steering, prev_velocity,
                  max_iters, goal_threshold, lookahead_distance, wheelbase, fov,
                  max_speed, rr, k_att, k_rep, step_size):
    # Coordinate system is y in the direction of the car and x perpendicular (right) to the car
    raw_scans_x = np.sin(fov) * obs['aaa_scans'][0]
    raw_scans_y = np.cos(fov) * obs['aaa_scans'][0]
    raw_scans_xy = np.vstack((raw_scans_x, raw_scans_y)).T

    # Filter the map_scans points that do not exceed the threshold distance from the others
    scans_xy = preprocess_scans(raw_scans_xy, threshold_gap=0.1, threshold_min=-4.0, threshold_max=10.0)

    obstacles = deepcopy(scans_xy)
    if scans_xy.shape[0] == 0:
        print('scans_xy empty')

    goals = find_target_point_all(scans_xy)
    goals = np.array(goals)

    if len(goals) == 0:
        # No gap was found
        # take the previous goal but move it closer to the car
        goal = np.clip(prev_goal - np.array([0.0, 0.5]), 0.0, None)
        print('goals empty')
    else:
        goal = goals[np.argmax(np.sqrt(np.sum(goals ** 2, axis=1)))]

    prev_goal = goal

    current_pos = np.array([0.0, 0.0])
    angle = np.pi / 2
    path = np.zeros((max_iters, 2))
    close_obstacles_ = None

    for i in range(1, max_iters):
        close_obstacles = find_close_obstacles(current_pos, obstacles)
        if close_obstacles_ is None:
            close_obstacles_ = close_obstacles
        current_pos, angle = step_potential_field(current_pos, angle, goal, close_obstacles, rr, k_att, k_rep,
                                                  step_size)
        path[i] = current_pos

        if distance_between_points(current_pos, goal) < goal_threshold:
            path = path[:i]
            break

    # Filter the path to make it smoother
    path_smooth = postprocess_path(path)

    distance_all_scans = scans_xy[np.sqrt(np.sum((scans_xy - np.array([0.0, 0.25])) ** 2, axis=1)) < 0.2, 1]

    if path_smooth.shape[0] < 2 or path_smooth[-1, 1] - 0.25 < 0.2:
        velocity = 0.0
        steering = prev_steering
        return steering, velocity, path_smooth, prev_steering, prev_goal, [0.0,
                                                                           0.0], scans_xy, close_obstacles_, raw_scans_xy
    elif len(distance_all_scans) > 0:
        min_dist = np.min(distance_all_scans)
        if min_dist < 0.25:
            velocity = 0.0
            steering = prev_steering
            return steering, velocity, [], prev_steering, prev_goal, [0.0,
                                                                      0.0], scans_xy, close_obstacles_, raw_scans_xy

    # Use a spline to calculate intermediate points with same distance
    xs = np.arange(path_smooth.shape[0])
    spline = CubicSpline(xs, path_smooth)
    x_smooth = np.linspace(min(xs), max(xs), 100)
    path_smooth = spline(x_smooth)

    position = np.array([0.0, 0.0])
    pose_theta = np.pi / 2
    max_reacquire = 20.0

    nearest_p, nearest_dist, t, i = nearest_point_on_trajectory(position, path_smooth[:, 0:2])
    if nearest_dist < lookahead_distance:
        lookahead_point, i2, t2 = first_point_on_trajectory_intersecting_circle(position,
                                                                                lookahead_distance,
                                                                                path_smooth[:, 0:2],
                                                                                i + t,
                                                                                wrap=True)
        if i2 is None:
            # return None, None
            lookahead_point, look_idx = path_smooth[-1, :], -1
        else:
            current_waypoint = np.array([path_smooth[i2, 0], path_smooth[i2, 1]])
            lookahead_point, look_idx = current_waypoint, i2
    elif nearest_dist < max_reacquire:
        lookahead_point, look_idx = path_smooth[i, :], i
    else:
        lookahead_point, look_idx = None, None

    if lookahead_point is None:
        lookahead_point = np.array([0.0, 0.0])
        print("Lookahead point is None")
        steering, speed = 0.0, 0.0
    else:
        # Take care that return args are flipped
        speed, steering = get_actuation(pose_theta,
                                        np.array([lookahead_point[0], lookahead_point[1], 0]),
                                        position,
                                        lookahead_distance,
                                        wheelbase)

    # Calculate velocity profile
    velocity = get_velocity_profile(goal, max_speed, steering)
    return steering, velocity, path_smooth, prev_steering, prev_goal, lookahead_point, scans_xy, close_obstacles_, raw_scans_xy


class PotentialFieldsPlanner:
    def __init__(self, config, vehicle_params=None):

        # Planner params
        self.lookahead_distance = config.lookahead_distance
        self.k_att = config.k_att
        self.k_rep = config.k_rep
        self.rr = config.rr
        self.step_size = np.array(config.step_size)
        self.max_iters = config.max_iters
        self.goal_threshold = config.goal_threshold
        self.target_l = config.target_l
        self.do_plot = config.do_plot

        # Vehicle params
        self.wheelbase = vehicle_params.lf + vehicle_params.lr
        self.max_speed = vehicle_params.v_max * config.vgain
        self.fov_deg = vehicle_params.fov_deg
        self.fov = np.deg2rad(np.linspace(self.fov_deg / 2, -self.fov_deg / 2, vehicle_params.lidar_points))

        self.figure = None

        self.prev_goal = np.array([0.0, 20.0])
        self.prev_steering = 0.0
        self.prev_velocity = 0.0

        self._plan = partial(
            plan_callback,
            max_iters=self.max_iters,
            goal_threshold=self.goal_threshold,
            lookahead_distance=self.lookahead_distance,
            wheelbase=self.wheelbase,
            fov=self.fov,
            max_speed=self.max_speed,
            rr=self.rr,
            k_att=self.k_att,
            k_rep=self.k_rep,
            step_size=self.step_size
        )
        self.start_time = time.time_ns()

    def plan(self, obs, waypoints=None):

        steering, velocity, path_smooth, prev_steering, prev_goal, lookahead_point, scans_xy, close_obstacles, raw_scans_xy = self._plan(
            obs, self.prev_goal, self.prev_steering, self.prev_velocity)

        self.prev_steering = steering
        self.prev_velocity = velocity
        self.prev_goal = prev_goal

        if self.do_plot:
            current_time = time.time_ns()
            if current_time - self.start_time > 1e6 / 24.0:
                self.plot(raw_scans_xy, path_smooth, prev_goal, lookahead_point, close_obstacles)
                self.start_time = current_time

        return steering, velocity

    def plot(self, scans_xy, path_smooth, prev_goal, lookahead_point, close_obstacles):
        # make a plt figure with equal axis
        if self.figure is None:
            self.figure = plt.figure(figsize=(10, 10))
            plt.axis('equal')
            plt.xlim(-3, 3)
            plt.ylim(-2, 5)
            plt.ion()
        if lookahead_point is not None:
            x_wp, y_wp = lookahead_point
        else:
            x_wp, y_wp = path_smooth[-1]
        plot_potential_fields(self.figure, scans_xy, close_obstacles, prev_goal[0], prev_goal[1], path_smooth, x_wp,
                              y_wp)

    def __del__(self):
        if self.figure is not None:
            plt.close(self.figure)

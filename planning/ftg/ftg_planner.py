import numpy as np
from planning.ftg.ftg_utils import pre_process_LiDAR, find_target_point, plot_lidar_scan


class FTGPlanner:
    """Planner that uses FTG to follow a trajectory."""

    def __init__(self, config_planner, vehicle_params):
        # Vehicle parameters from the car model
        self.max_speed = vehicle_params.v_max

        # Lidar parameters
        self.angle_incre = np.deg2rad(vehicle_params.fov_deg / vehicle_params.lidar_points)  # 0.00435
        self.angle_min = -np.deg2rad(vehicle_params.fov_deg / 2)  # -2.35
        self.mu = vehicle_params.mu
        self.wheelbase = vehicle_params.lf + vehicle_params.lr

        # Config for the planner
        self.vgain = config_planner.vgain
        self.safe_vgain = config_planner.safe_vgain
        self.P = config_planner.P

        # Scan filtering parameters
        self.window_size = config_planner.window_size
        self.safe_thres = config_planner.safe_thres
        self.danger_thres = config_planner.danger_thres
        self.rb = config_planner.rb
        self.max_gap_length = config_planner.max_gap_length
        self.min_gap_length = config_planner.min_gap_length

        # Visualization options
        self.do_plot = config_planner.do_plot

        self.prev_steering = 0.0
        self.prev_velocity = 0.0

    def filter_scan(self, scan):
        return pre_process_LiDAR(scan, self.window_size, self.danger_thres, self.rb)

    def get_target_point(self, scan):
        return find_target_point(scan, self.safe_thres, self.max_gap_length, self.min_gap_length)

    def plan(self, obs, waypoints=None):
        scan = obs['aaa_scans'][0]
        filtered_scan = self.filter_scan(scan)
        best_p_idx = self.get_target_point(filtered_scan)

        if best_p_idx is not None:
            steering = (self.angle_min + best_p_idx * self.angle_incre)  * self.P
            steering = (self.angle_min + best_p_idx * self.angle_incre)  * self.P
            steering = self.prev_steering + np.clip(steering - self.prev_steering, -0.2, 0.2)
            velocity_target = self.max_speed
        else:
            steering = obs['steering_angle'][0]
            velocity_target = 4.0

        if best_p_idx is not None and np.min(scan[best_p_idx - 10:best_p_idx + 10]) < 2.0:
            velocity_steering = 4.0
        else:
            velocity_steering = self.max_speed

        velocity_safe = np.sqrt(self.mu * 9.81 / (np.tan(np.abs(steering + 1e-5)) / self.wheelbase)) * self.safe_vgain
        velocity = min(velocity_target, velocity_safe, velocity_steering) * self.vgain

        self.prev_steering = steering
        self.prev_velocity = velocity

        if self.do_plot:
            plot_lidar_scan(filtered_scan, best_p_idx)

        return steering, velocity

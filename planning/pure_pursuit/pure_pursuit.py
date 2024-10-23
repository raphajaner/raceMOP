import numpy as np
from f110_gym.envs.env_utils import *
from planning.planner_base import PlannerBase


class PurePursuitPlanner(PlannerBase):
    """Planner that uses pure pursuit to follow a trajectory."""

    def __init__(self, config, vehicle_params):
        super().__init__(config)
        # Config for the planner
        self.config = config
        self.tlad = config.tlad
        self.vgain = config.vgain
        self.vmax = config.vmax
        self.n_next_points = config.n_next_points
        self.skip_next_points = config.skip_next_points

        # Vehicle parameters from the car model
        self.vehicle_params = vehicle_params
        self.wheelbase = vehicle_params.lf + vehicle_params.lr

        self.max_reacquire = 20.
        self.drawn_waypoints = []

    def plan(self, obs, waypoints):
        """Plan a trajectory to follow the waypoints.

        Args:
            obs (dict): observation dict
            waypoints (np.ndarray): waypoints to follow
        Returns:
            steering (float): steering angle
            velocity (float): velocity
            lookahead_points_relative (np.ndarray): relative lookahead points
        """
        poses_global = np.array([obs['poses_x'][0], obs['poses_y'][0], obs['poses_theta'][0]], dtype=np.float64)

        waypoints = get_current_waypoint(
            waypoints,
            20,
            self.config.tlad,
            np.array([obs['poses_x'][0], obs['poses_y'][0]]),
            n_next_points=2  # self.config.n_next_points
        )

        if waypoints is None:
            return 0.0, 0.0

        speed, steering = get_actuation(
            poses_global[2], waypoints[0, [0, 1, 3]], poses_global[:2], self.tlad, self.wheelbase
        )

        velocity = self.vgain * speed

        if self.vmax != 'None':
            velocity = np.clip(velocity, 0, self.vmax)

        return steering, velocity

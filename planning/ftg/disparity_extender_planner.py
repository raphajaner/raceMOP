import numpy as np
from numba import njit

from planning.ftg.ftg_utils import pre_process_LiDAR_disparity
from planning.ftg.ftg_planner import FTGPlanner


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


class DisparityExtenderPlanner(FTGPlanner):
    """Planner that uses the FTG planner to follow a trajectory extended by disparity information."""

    def __init__(self, config_planner, vehicle_params):
        super().__init__(config_planner, vehicle_params)
        self.prev_goal = np.array([0.0, 20.0])

    def filter_scan(self, scan):
        return pre_process_LiDAR_disparity(scan)

    def get_target_point(self, scan):
        fov = np.linspace(4.7 / 2, -4.7 / 2.0, 1080)
        x_map_scans = np.sin(fov) * scan
        y_map_scans = np.cos(fov) * scan
        raw_scans_xy = np.stack([x_map_scans, y_map_scans], axis=1)

        goals = find_target_point_all(raw_scans_xy)

        if len(goals) == 0:
            # No gap was found
            goal = np.clip(self.prev_goal - np.array([0.0, 0.5]), 0.0, None)
        else:
            best_value = -1
            best_idx = None
            for goal in goals:
                closest_point_idx = np.argmin(np.sqrt(np.sum((raw_scans_xy - goal) ** 2, axis=1)))
                closest_point_idx = np.clip(closest_point_idx, 180, 900)
                value = scan[-10 + closest_point_idx:closest_point_idx + 10].mean()
                if value > best_value:
                    best_value = value
                    best_idx = closest_point_idx
            # goal = goals[np.argmax(np.sqrt(np.sum((goals - np.array([0.0, 0.0])) ** 2, axis=1)))]

        self.prev_goal = goal
        return best_idx

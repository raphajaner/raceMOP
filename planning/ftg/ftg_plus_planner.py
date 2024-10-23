import numpy as np
from planning.ftg.ftg_planner import FTGPlanner


def find_target_point(ranges, safe_thres, max_gap_length=290, min_gap_length=10):
    """_summary_
        Find all the gaps exceed a safe thres.
        Among those qualified gaps, chose the one with a farmost point, calculate the target as the middle of the gap.
    Args:
        ranges (_type_): _description_
        safe_thres (_type_): _description_
        max_gap_length (int, optional): _description_. Defaults to 350.
    Returns:
        target: int
    """
    n = len(ranges)
    safe_p_left, safe_p_right = 0, n - 1
    p = safe_p_left
    safe_range = []
    fov = np.linspace(4.7 / 2, -4.7 / 2.0, 1080)
    while p < n - 1:
        if ranges[p] >= safe_thres:
            safe_p_left = p
            p += 1
            while p < n - 1 and ranges[p] >= safe_thres and p - safe_p_left <= max_gap_length:
                p += 1
            safe_p_right = p - 1
            if safe_p_right != safe_p_left:
                safe_range.append((-abs(-safe_p_right + safe_p_left), (safe_p_left, safe_p_right)))
        else:
            p += 1

    if len(safe_range) == 0:
        print('no safe rang 2323232')
        return np.argmax(ranges)
    else:
        while not len(safe_range) == 0:
            min_i = 0
            for i in range(len(safe_range)):
                if safe_range[i][0] < safe_range[min_i][0]:
                    min_i = i
            item = safe_range[min_i]
            safe_p_left, safe_p_right = item[1]
            del safe_range[min_i]
            if abs(safe_p_right - safe_p_left) > min_gap_length:
                safe_p_left, safe_p_right = item[1]
                angle1 = fov[safe_p_left]
                angle2 = fov[safe_p_right]
                angle = (angle1 + angle2) / 2.0
                # find idx closest to angle
                target = np.argmin(np.abs(fov - angle))
                print(f'target: {target}')
                if 190 < target < 900:
                    return target


class FTGPlusPlanner(FTGPlanner):
    """FTG planner with additional features."""
    def get_target_point(self, scan):
        return find_target_point(scan, self.safe_thres, self.max_gap_length, self.min_gap_length)

import numpy as np
from matplotlib import patches
from numba import njit
import matplotlib.pyplot as plt
from queue import PriorityQueue


@njit(fastmath=False, cache=True)
def pre_process_LiDAR(ranges, window_size, danger_thres, rb):
    # roughly filter with mean
    proc_ranges = []
    for i in range(0, len(ranges), window_size):
        cur_mean = sum(ranges[i:i + window_size]) / window_size
        for _ in range(window_size):
            proc_ranges.append(cur_mean)
    proc_ranges = np.array(proc_ranges)
    # set danger range and ranges too far to zero
    p, n = 0, len(proc_ranges)
    while p < n:
        if proc_ranges[p] <= danger_thres:
            ranges[max(0, p - rb): p + rb] = 0
            p += rb
        else:
            p += 1
    return proc_ranges


@njit(fastmath=False, cache=True)
def pre_process_LiDAR_disparity(ranges):
    lidar_samples = ranges
    threshold = 2.0
    car_width = 0.3
    tolerance = 0.0

    angular_resolution = 270 / len(lidar_samples)  # Degrees per sample
    # Set distance to 0 for all points outside -90 to +90 degrees
    # angle_range = int(90 / angular_resolution)
    # center = len(lidar_samples) // 2
    # lidar_samples[:center - angle_range] = 0
    # lidar_samples[center + angle_range:] = 0

    # Identify disparities between subsequent points
    disparities = np.abs(np.diff(lidar_samples))
    disparities_idx = np.where(disparities > threshold)[0]

    for i in disparities_idx:
        # Determine the more distant and the closer point
        if lidar_samples[i] >= lidar_samples[i + 1]:
            more_distant_index = i
            closer_distance = lidar_samples[i + 1]
            further_distance = lidar_samples[i]
            direction = -1  # Overwrite in the reverse direction
        else:
            more_distant_index = i + 1
            closer_distance = lidar_samples[i]
            further_distance = lidar_samples[i + 1]
            direction = 1  # Overwrite in the forward direction

        if closer_distance > 0.0:  # Avoid divide by zero
            # Calculate the angular width of the car at the closer distance
            angular_width = np.arctan((car_width / 2 + tolerance) / closer_distance) * (180 / np.pi) * 2
            num_samples_to_overwrite = int(np.ceil(angular_width / angular_resolution)) + 20

            # Overwrite the LIDAR samples starting from the more distant point
            for j in range(-1, num_samples_to_overwrite):
                index_to_overwrite = more_distant_index + j * direction
                if 0 <= index_to_overwrite < len(lidar_samples):
                    if lidar_samples[index_to_overwrite] > closer_distance:
                        lidar_samples[index_to_overwrite] = closer_distance

    return lidar_samples


def find_target_point(ranges, safe_thres, max_gap_length=900, min_gap_length=30):
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
    safe_range = PriorityQueue()
    while p < n - 1:
        if ranges[p] >= safe_thres:
            safe_p_left = p
            p += 1
            # while p < end_i and ranges[p] >= self.safe_thres and p-safe_p_left <= 290:
            # import pdb
            # pdb.set_trace()
            while p < n - 1 and ranges[p] >= safe_thres and p - safe_p_left <= max_gap_length:  # and p < safe_p_right:
                p += 1
            safe_p_right = p - 1
            if safe_p_right != safe_p_left:
                # check that points between are all safe, i.e., further away than the gap end points
                # import pdb
                # pdb.set_trace()
                # if np.all(ranges[safe_p_left + 1:safe_p_right] >= ranges[safe_p_left]) or np.all(
                #         ranges[safe_p_left + 1:safe_p_right] >= ranges[safe_p_right]):
                # higher priority for larger gap
                # safe_range.put((-safe_p_right + safe_p_left, (safe_p_left, safe_p_right)))
                # PrioQueue retrieves the smallest item first

                fov = np.linspace(4.7 / 2, -4.7 / 2.0, 1080)
                angle1 = fov[safe_p_left]
                angle2 = fov[safe_p_right]
                angle = (angle1 + angle2) / 2.0
                target = np.argmin(np.abs(fov - angle))

                # Get the angle of the opening
                # angle = np.abs(angle1 - angle2)

                if target < 190 or target > 900:
                    continue

                try:
                    # priority = -np.min(ranges[target-10:target+10])
                    priority = -np.abs(angle1 - angle2)
                except:
                    import pdb
                    pdb.set_trace()

                safe_range.put(((priority), (safe_p_left, safe_p_right)))
                # safe_range.put(((-np.min(ranges[safe_p_left:safe_p_right])), (safe_p_left, safe_p_right)))

        else:
            p += 1

    # print len of queue
    print(safe_range.qsize())
    # if safe_range.empty():
    #     # print('no safe range')
    #     return np.argmax(ranges)
    # else:
    while not safe_range.empty():
        # import pdb
        # pdb.set_trace()
        safe_p_left, safe_p_right = safe_range.get(block=False)[1]
        if safe_p_right - safe_p_left > min_gap_length:
            # target = (safe_p_left + safe_p_right) // 2

            fov = np.linspace(4.7 / 2, -4.7 / 2.0, 1080)
            angle1 = fov[safe_p_left]
            angle2 = fov[safe_p_right]
            angle = (angle1 + angle2) / 2.0
            target = np.argmin(np.abs(fov - angle))

            # print(angle)
            # target -= 30
            if 190 < target < 900:  # and dist > 3.0: #and angle > np.deg2rad(10):
                return target


def plot_lidar_scan(scan, best_p_idx):
    ax = plt.gca()
    fov = [angle for angle in np.linspace(4.7 / 2, -4.7 / 2.0, 1080)]

    plt.cla()
    plt.axis('equal')
    ax.grid(True)

    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 11)
    x_map_scans = [np.sin(angle) * length for angle, length in zip(fov, scan)]
    y_map_scans = [np.cos(angle) * length for angle, length in zip(fov, scan)]

    if best_p_idx is not None:
        target_x = x_map_scans[best_p_idx]
        target_y = y_map_scans[best_p_idx]
        ax.plot([0, target_x], [0, target_y], color="red")
        plt.scatter(target_x, target_y, color="green", marker="*", s=10, linewidths=3)

    # cmax_p_idx = np.argmax(scan)
    # target_l = scan[cmax_p_idx] - 1
    # target_x = x_map_scans2[best_p_idx]
    # target_y = y_map_scans2[best_p_idx]
    # target_l = scan[best_p_idx]

    # x_map_scans = [np.sin(angle) * length if length < target_l else 0 for angle, length in zip(fov, scan)]
    # y_map_scans = [np.cos(angle) * length if length < target_l else 0 for angle, length in zip(fov, scan)]
    ax.add_patch(patches.Rectangle((-0.15, -0.25), 0.3, 0.5, color="black"))

    plt.scatter(x_map_scans, y_map_scans, color="blue", s=1)

    #
    # raw_scans_x = np.sin(fov) * obs['aaa_scans'][0]
    # raw_scans_y = np.cos(fov) * obs['aaa_scans'][0]
    # ax.scatter(raw_scans_x, raw_scans_y, color="blue", s=0.5)
    #
    # # add target point
    # target_x = np.sin(fov[best_p_idx]) * obs['aaa_scans'][0][best_p_idx]
    # target_y = np.cos(fov[best_p_idx]) * obs['aaa_scans'][0][best_p_idx]
    # ax.scatter(target_x, target_y, color="green", s=4)

    plt.pause(0.000000001)

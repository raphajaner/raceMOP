from copy import deepcopy
import numpy as np
from numba import njit
import os
from PIL import Image
import yaml
from scipy.ndimage import distance_transform_edt as edt


class CircularQueue:
    def __init__(self, size):
        self.size = size
        self.queue = np.ones(size)
        self.pointer = 0

    def add(self, element):
        self.queue[self.pointer] = element
        self.pointer = (self.pointer + 1) % self.size

    def mean(self):
        return np.mean(self.queue)

    def max(self):
        return np.max(self.queue)


class Map:
    def __init__(self, map_path, map_ext):
        map_img_path = os.path.splitext(map_path)[0] + map_ext
        self.map_img = np.array(Image.open(map_img_path).transpose(Image.FLIP_TOP_BOTTOM)).astype(np.float64)
        self.map_img[self.map_img <= 128.] = 0.
        self.map_img[self.map_img > 128.] = 255.
        self.map_height = self.map_img.shape[0]
        self.map_width = self.map_img.shape[1]
        # load map yaml
        with open(map_path + '.yaml', 'r') as yaml_stream:
            try:
                map_metadata = yaml.safe_load(yaml_stream)
                self.map_resolution = map_metadata['resolution']
                self.origin = map_metadata['origin']
            except yaml.YAMLError as ex:
                print(ex)

        self.orig_x = self.origin[0]
        self.orig_y = self.origin[1]
        self.orig_s = np.sin(self.origin[2])
        self.orig_c = np.cos(self.origin[2])

        self.dt = self.map_resolution * edt(self.map_img)
        self.map_metainfo = (
            self.orig_x, self.orig_y, self.orig_c, self.orig_s, self.map_height, self.map_width,
            self.map_resolution)


@njit
def get_trackline_segment(path, point):
    # replace without using norm function
    dists = np.sqrt((point[0] - path[:, 0]) ** 2 + (point[1] - path[:, 1]) ** 2)
    min_dist_segment = np.argmin(dists)
    if min_dist_segment == 0:
        return 0, dists
    elif min_dist_segment == len(dists) - 1:
        return len(dists) - 2, dists

    if dists[min_dist_segment + 1] < dists[min_dist_segment - 1]:
        return min_dist_segment, dists
    else:
        return min_dist_segment - 1, dists


def interp_pts(ss, idx, dists):
    """
    Returns the distance along the trackline and the height above the trackline
    Finds the reflected distance along the line joining wpt1 and wpt2
    Uses Herons formula for the area of a triangle

    """
    d_ss = ss[idx + 1] - ss[idx]
    d1, d2 = dists[idx], dists[idx + 1]

    if d1 < 0.01:  # at the first point
        x = 0
        h = 0
    elif d2 < 0.01:  # at the second point
        x = dists[idx]  # the distance to the previous point
        h = 0  # there is no distance
    else:
        # if the point is somewhere along the line
        s = (d_ss + d1 + d2) / 2
        Area_square = (s * (s - d1) * (s - d2) * (s - d_ss))
        Area = Area_square ** 0.5
        h = Area * 2 / d_ss
        if np.isnan(h):
            h = 0
        x = (d1 ** 2 - h ** 2) ** 0.5

    return x, h


class TrackLine:
    # Taken from https://github.com/BDEvan5/F1TenthRacingDRL/blob/master/F1TenthRacingDRL/Planners/TrackLine.py#L208

    def __init__(self, map_name, map_paths, config_map, filename):
        self.config_map = config_map
        self.map_name = map_name
        self.map_paths = map_paths

        self.path = None
        self.theta = None
        self.velocity = None
        self.kappa = None
        self.s_track = None
        self.ss = None

        track = np.loadtxt(filename, delimiter=self.config_map.wpt_delim, skiprows=self.config_map.wpt_rowskip)

        self.path = track[:, [self.config_map.wpt_xind, self.config_map.wpt_yind]]
        self.theta = track[:, self.config_map.wpt_thind]
        try:
            self.velocity = track[:, self.config_map.wpt_vind]
        except:
            self.velocity = np.zeros_like(self.path[:, 0])
        self.kappa = track[:, self.config_map.wpt_kappa]
        self.s_track = track[:, 0]

        seg_lengths = np.linalg.norm(np.diff(self.path, axis=0), axis=1)
        self.ss = np.insert(np.cumsum(seg_lengths), 0, 0)

        self.total_s = self.ss[-1]
        self.N = len(self.path)

        self.diffs = self.path[1:, :] - self.path[:-1, :]
        self.l2s = self.diffs[:, 0] ** 2 + self.diffs[:, 1] ** 2

        self.max_distance = 0
        self.distance_allowance = 1

        # map itself
        self.map = Map(map_paths, map_ext=config_map.map_ext)

    @property
    def waypoints(self):
        # x, y, theta, kappa, v
        return deepcopy(np.vstack((self.path[:, 0], self.path[:, 1], self.theta, self.kappa, self.velocity)).T)

    def calculate_progress(self, point):
        idx, dists = self.get_trackline_segment(point)

        x, h = self.interp_pts(idx, dists)

        s = self.ss[idx] + x

        return s

    def calculate_progress_percent(self, point):
        s = self.calculate_progress(point)
        s_percent = s / self.total_s

        return s_percent

    def interp_pts(self, idx, dists):
        """
        Returns the distance along the trackline and the height above the trackline
        Finds the reflected distance along the line joining wpt1 and wpt2
        Uses Herons formula for the area of a triangle

        """
        return interp_pts(self.ss, idx, dists)

    def get_trackline_segment(self, point):
        return get_trackline_segment(self.path, point)


class RacingLine(TrackLine):
    def __init__(self, map_name, map_paths, config_map):
        filename = config_map.map_path + f'{map_name}/{map_name}_raceline.csv'
        super().__init__(map_name, map_paths, config_map, filename)


class CenterLine(TrackLine):
    def __init__(self, map_name, map_paths, config_map):
        filename = config_map.map_path + f'{map_name}/{map_name}_centerline_vel_newconv.csv'
        super().__init__(map_name, map_paths, config_map, filename)

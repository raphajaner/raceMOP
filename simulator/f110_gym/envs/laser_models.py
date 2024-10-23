import os

import numpy as np
import yaml
from PIL import Image
from numba import njit, prange
from scipy.ndimage import distance_transform_edt as edt

from f110_gym.envs.env_utils import xy_2_rc, cross, are_collinear


def get_dt(bitmap, resolution):
    """ Distance transformation of the input bitmap.

    Uses scipy.ndimage, cannot be JITted.

    Args:
        - bitmap (numpy.ndarray, (n, m)): input binary bitmap of the environment, where 0 is obstacles, and 255 (or
            anything > 0) is freespace
        - resolution (float): resolution of the input bitmap (m/cell)

    Returns: dt (numpy.ndarray, (n, m)): output distance matrix, where each cell has the corresponding distance (in
    meters) to the closest obstacle
    """
    dt = resolution * edt(bitmap)
    return dt


@njit(cache=True)
def distance_transform(x, y, orig_x, orig_y, orig_c, orig_s, height, width, resolution, dt):
    """Look up corresponding distance in the distance matrix.

    Args:
        x (float): x coordinate of the point (m)
        y (float): y coordinate of the point (m)
        orig_x (float): x coordinate of the map origin (m)
        orig_y (float): y coordinate of the map origin (m)
        orig_c (float): cosine of the map origin rotation
        orig_s (float): sine of the map origin rotation
        height (int): height of the map in cells
        width (int): width of the map in cells
        resolution (float): resolution of the map in meters/cell
        dt (numpy.ndarray (height, width)): distance transform matrix

    Returns:
        distance (float): distance to the nearest obstacle from the point (x, y)
    """

    r, c = xy_2_rc(x, y, orig_x, orig_y, orig_c, orig_s, height, width, resolution)
    distance = dt[r, c]
    return distance


@njit(cache=True)
def trace_ray(x, y, theta_index, sines, cosines, eps, orig_x, orig_y, orig_c, orig_s, height, width, resolution, dt,
              max_range):
    """ Find the length of a specific ray at a specific scan angle theta."""

    # int casting, and index precal trigs
    theta_index_ = int(theta_index)
    s = sines[theta_index_]
    c = cosines[theta_index_]

    # distance to nearest initialization
    dist_to_nearest = distance_transform(x, y, orig_x, orig_y, orig_c, orig_s, height, width, resolution, dt)
    total_dist = dist_to_nearest

    # ray tracing iterations
    while dist_to_nearest > eps and total_dist <= max_range:
        # move in the direction of the ray by dist_to_nearest
        x += dist_to_nearest * c
        y += dist_to_nearest * s

        # update dist_to_nearest for current point on ray
        # also keeps track of total ray length
        dist_to_nearest = distance_transform(x, y, orig_x, orig_y, orig_c, orig_s, height, width, resolution, dt)
        total_dist += dist_to_nearest

    if total_dist > max_range:
        total_dist = max_range

    return total_dist


@njit(cache=True)
def get_scan(pose, theta_dis, fov, num_beams, theta_index_increment, sines, cosines, eps, orig_x, orig_y, orig_c,
             orig_s, height, width, resolution, dt, max_range):
    """Perform the scan for each discrete angle of each beam of the laser.

    Loop heavy, should be JITted

    Args:
        pose (numpy.ndarray(3, )): current pose of the scan frame in the map
        theta_dis (int): number of steps to discretize the angles between 0 and 2pi for look up
        fov (float): field of view of the laser scan
        num_beams (int): number of beams in the scan
        theta_index_increment (float): increment between angle indices after discretization
        sines (numpy.ndarray (n, )): pre-calculated sines of the angle array
        cosines (numpy.ndarray (n, )): pre-calculated cosines ...
        eps (float): epsilon for distance transform
        orig_x (float): x coordinate of the map origin (m)
        orig_y (float): y coordinate of the map origin (m)
        orig_c (float): cosine of the map origin rotation
        orig_s (float): sine of the map origin rotation
        height (int): height of the map in cells
        width (int): width of the map in cells
        resolution (float): resolution of the map in meters/cell
        dt (numpy.ndarray (height, width)): distance transform matrix
        max_range (float): maximum range of the laser

    Returns:
        scan (numpy.ndarray(n, )): resulting laser scan at the pose, n=num_beams
    """
    # empty scan array init
    scan = np.empty((num_beams,))

    # make theta discrete by mapping the range [-pi, pi] onto [0, theta_dis]
    theta_index = theta_dis * (pose[2] - fov / 2.) / (2. * np.pi)
    # make sure it's wrapped properly
    theta_index = np.fmod(theta_index, theta_dis)
    while theta_index < 0:
        theta_index += theta_dis

    theta_index_ = theta_index

    # # sweep through each beam
    # for i in prange(0, num_beams):
    #     # trace the current beam
    #     scan[i] = trace_ray(pose[0], pose[1], theta_index, sines, cosines, eps, orig_x, orig_y, orig_c, orig_s, height,
    #                         width, resolution, dt, max_range)
    #
    #     # increment the beam index
    #     theta_index += theta_index_increment
    #
    #     # make sure it stays in the range [0, theta_dis)
    #     while theta_index >= theta_dis:
    #         # print('used to be theta_index -= theta_dis')
    #         theta_index -= theta_dis

    scan2 = np.empty((num_beams,))
    # precalculate theta_index
    # theta_index += theta_index_increment
    theta_index2 = np.zeros((num_beams,))
    theta_index2[0] = theta_index_
    for i in range(1, num_beams):
        # increment the beam index
        theta_index2[i] = theta_index2[i - 1] + theta_index_increment

        # make sure it stays in the range [0, theta_dis)
        while theta_index2[i] >= theta_dis:
            # print('used to be theta_index -= theta_dis')
            theta_index2[i] -= theta_dis

    # make sure it stays in the range [0, theta_dis)
    # theta_index = np.fmod(theta_index, theta_dis)
    # while theta_index < 0:

    for i in prange(0, num_beams):
        # trace the current beam
        scan2[i] = trace_ray(
            pose[0], pose[1], theta_index2[i], sines, cosines, eps, orig_x, orig_y, orig_c, orig_s, height, width,
            resolution, dt, max_range
        )
    # check if scan and scan2 are the same
    scan = scan2

    return scan


@njit(cache=True)
def check_ttc_jit(scan, vel, scan_angles, cosines, side_distances, ttc_thresh):
    """Checks the iTTC of each beam in a scan for collision with environment.

    Args:
        scan (np.ndarray(num_beams, )): current scan to check
        vel (float): current velocity
        scan_angles (np.ndarray(num_beams, )): precomped angles of each beam
        cosines (np.ndarray(num_beams, )): precomped cosines of the scan angles
        side_distances (np.ndarray(num_beams, )): precomped distances at each beam from the laser to the sides of the car
        ttc_thresh (float): threshold for iTTC for collision

    Returns:
        in_collision (bool): whether vehicle is in collision with environment
        collision_angle (float): at which angle the collision happened
    """
    in_collision = False
    if vel != 0.0:
        num_beams = scan.shape[0]
        for i in range(num_beams):
            proj_vel = vel * cosines[i]
            ttc = (scan[i] - side_distances[i]) / proj_vel
            if (ttc < ttc_thresh) and (ttc >= 0.0):
                in_collision = True
                break
    return in_collision


@njit(cache=True)
def get_range(pose, beam_theta, va, vb):
    """Get the distance at a beam angle to the vector formed by two of the four vertices of a vehicle.

    Args:
        pose (np.ndarray(3, )): pose of the scanning vehicle
        beam_theta (float): angle of the current beam (world frame)
        va, vb (np.ndarray(2, )): the two vertices forming an edge

    Returns:
        distance (float): smallest distance at beam theta from scanning pose to edge
    """
    o = pose[0:2]
    v1 = o - va
    v2 = vb - va
    v3 = np.array([np.cos(beam_theta + np.pi / 2.), np.sin(beam_theta + np.pi / 2.)])

    denom = v2.dot(v3)
    distance = np.inf

    if np.fabs(denom) > 0.0:
        d1 = cross(v2, v1) / denom
        d2 = v1.dot(v3) / denom
        if d1 >= 0.0 and 0.0 <= d2 <= 1.0:
            distance = d1
    elif are_collinear(o, va, vb):
        da = np.linalg.norm(va - o)
        db = np.linalg.norm(vb - o)
        distance = min(da, db)

    return distance


@njit(cache=True)
def get_blocked_view_indices(pose, vertices, scan_angles):
    """Get the indices of the start and end of blocked fov in scans by another vehicle.

    Args:
        pose (np.ndarray(3, )): pose of the scanning vehicle
        vertices (np.ndarray(4, 2)): four vertices of a vehicle pose
        scan_angles (np.ndarray(num_beams, )): corresponding beam angles
    """
    # find four vectors formed by pose and 4 vertices:
    vecs = vertices - pose[:2]
    vec_sq = np.square(vecs)
    norms = np.sqrt(vec_sq[:, 0] + vec_sq[:, 1])
    unit_vecs = vecs / norms.reshape(norms.shape[0], 1)

    # find angles between all four and pose vector
    ego_x_vec = np.array([[np.cos(pose[2])], [np.sin(pose[2])]])

    angles_with_x = np.empty((4,))
    for i in range(4):
        angle = np.arctan2(ego_x_vec[1], ego_x_vec[0]) - np.arctan2(unit_vecs[i, 1], unit_vecs[i, 0])
        if angle > np.pi:
            angle = angle - 2 * np.pi
        elif angle < -np.pi:
            angle = angle + 2 * np.pi
        angles_with_x[i] = -angle[0]

    ind1 = int(np.argmin(np.abs(scan_angles - angles_with_x[0])))
    ind2 = int(np.argmin(np.abs(scan_angles - angles_with_x[1])))
    ind3 = int(np.argmin(np.abs(scan_angles - angles_with_x[2])))
    ind4 = int(np.argmin(np.abs(scan_angles - angles_with_x[3])))
    indices = [ind1, ind2, ind3, ind4]
    return min(indices), max(indices)


@njit(cache=True)
def ray_cast(pose, scan, scan_angles, vertices):
    """Modify a scan by ray casting onto another agents.py's four vertices.

    Args:
        pose (np.ndarray(3, )): pose of the vehicle performing scan
        scan (np.ndarray(num_beams, )): original scan to modify
        scan_angles (np.ndarray(num_beams, )): corresponding beam angles
        vertices (np.ndarray(4, 2)): four vertices of a vehicle pose
    
    Returns:
        new_scan (np.ndarray(num_beams, )): modified scan
    """
    # pad vertices so loops around
    looped_vertices = np.empty((5, 2))
    looped_vertices[0:4, :] = vertices
    looped_vertices[4, :] = vertices[0, :]

    min_ind, max_ind = get_blocked_view_indices(pose, vertices, scan_angles)
    # looping over beams
    for i in range(min_ind, max_ind + 1):
        # looping over vertices
        for j in range(4):
            # check if original scan is longer than ray casted distance
            scan_range = get_range(pose, pose[2] + scan_angles[i], looped_vertices[j, :], looped_vertices[j + 1, :])
            if scan_range < scan[i]:
                scan[i] = scan_range
    return scan


class ScanSimulator2D(object):
    """2D LIDAR scan simulator class.

    Attributes:
        num_beams (int): number of beams in the scan
        fov (float): field of view of the laser scan
        eps (float, default=0.0001): ray tracing iteration termination condition
        theta_dis (int, default=2000): number of steps to discretize the angles between 0 and 2pi for look up
        max_range (float, default=30.0): maximum range of the laser
    """

    def __init__(self, num_beams, fov, eps=0.0001, theta_dis=2000, max_range=30.0):
        # initialization 
        self.num_beams = num_beams
        self.fov = fov
        self.eps = eps
        self.theta_dis = theta_dis
        self.max_range = max_range
        self.angle_increment = self.fov / (self.num_beams - 1)
        self.theta_index_increment = theta_dis * self.angle_increment / (2. * np.pi)
        self.orig_c = None
        self.orig_s = None
        self.orig_x = None
        self.orig_y = None
        self.map_height = None
        self.map_width = None
        self.map_resolution = None
        self.dt = None

        # precomputing corresponding cosines and sines of the angle array
        theta_arr = np.linspace(0.0, 2 * np.pi, num=theta_dis)
        self.sines = np.sin(theta_arr)
        self.cosines = np.cos(theta_arr)

    def set_map(self, map_path, map_ext):
        """ Set the bitmap of the scan simulator by path.

        Args:
            map_path (str): path to the map yaml file
            map_ext (str): extension (image type) of the map image

        Returns:
            flag (bool): if image reading and loading is successful
        """

        # load map image
        map_img_path = os.path.splitext(map_path)[0] + map_ext
        self.map_img = np.array(Image.open(map_img_path).transpose(Image.FLIP_TOP_BOTTOM))
        self.map_img = self.map_img.astype(np.float64)

        # grayscale -> binary
        self.map_img[self.map_img <= 128.] = 0.
        self.map_img[self.map_img > 128.] = 255.

        self.map_height = self.map_img.shape[0]
        self.map_width = self.map_img.shape[1]

        # load map yaml
        with open(map_path, 'r') as yaml_stream:
            try:
                map_metadata = yaml.safe_load(yaml_stream)
                self.map_resolution = map_metadata['resolution']
                self.origin = map_metadata['origin']
            except yaml.YAMLError as ex:
                print(ex)

        # calculate map parameters
        self.orig_x = self.origin[0]
        self.orig_y = self.origin[1]
        self.orig_s = np.sin(self.origin[2])
        self.orig_c = np.cos(self.origin[2])

        # get the distance transform
        self.dt = get_dt(self.map_img, self.map_resolution)
        return True

    def scan(self, pose, rng, std_dev=0.01):
        """Perform simulated 2D scan by pose on the given map.

        Args:
            pose (np.ndarray(3, )): current pose of the scan frame in the map
            rng (np.random.RandomState): random number generator
            std_dev (float, default=0.01): standard deviation of the noise to add to the scan

        Returns:
            scan (np.ndarray(num_beams, )): resulting laser scan at the pose
        """

        assert self.map_height is not None, 'Map is not set for scan simulator.'

        scan = get_scan(pose, self.theta_dis, self.fov, self.num_beams, self.theta_index_increment, self.sines,
                        self.cosines, self.eps, self.orig_x, self.orig_y, self.orig_c, self.orig_s, self.map_height,
                        self.map_width, self.map_resolution, self.dt, self.max_range)

        if rng is not None:
            noise = rng.normal(0., std_dev, size=self.num_beams)
            scan += noise

        return scan

    def get_increment(self):
        return self.angle_increment

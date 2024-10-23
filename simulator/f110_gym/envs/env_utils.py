import numpy as np
from numba import njit


@njit(fastmath=False, cache=True)
def nearest_point_on_trajectory(point, trajectory):
    """Return the nearest point along the given piecewise linear trajectory.

    Note: Trajectories must be unique. If they are not unique, a divide by 0 error will occur

    Args:
        point(np.ndarray): size 2 numpy array
        trajectory: Nx2 matrix of (x,y) trajectory waypoints

    Returns:
        projection(np.ndarray): size 2 numpy array of the nearest point on the trajectory
        dist(float): distance from the point to the projection
        t(float): the t value of the projection along the trajectory
        min_dist_segment(int): the index of the segment of the trajectory that the projection is on
    """
    diffs = trajectory[1:, :] - trajectory[:-1, :]
    # if a point es exactly on a waypoint, return that waypoint
    distances = np.sqrt(np.sum((trajectory - point) ** 2, axis=1))
    min_dist_segment = np.argmin(distances)
    if distances[min_dist_segment] < 1e-6:
        return trajectory[min_dist_segment], np.linalg.norm(trajectory[min_dist_segment] - point), 0.0, min_dist_segment

    # Equivalent to dot product
    t = np.sum((point - trajectory[:-1, :]) * diffs, axis=1) / (diffs[:, 0] ** 2 + diffs[:, 1] ** 2)
    t = np.clip(t, 0.0, 1.0)
    projections = trajectory[:-1, :] + (t * diffs.T).T
    temp = point - projections
    dists = np.sqrt(np.sum(temp * temp, axis=1))
    min_dist_segment = np.argmin(dists)
    return projections[min_dist_segment], dists[min_dist_segment], t[min_dist_segment], min_dist_segment


@njit(fastmath=False, cache=True)
def first_point_on_trajectory_intersecting_circle(point, radius, trajectory, t=0.0, wrap=False):
    """Return the first point along the given piecewise linear trajectory that intersects the given circle.

    Assumes that the first segment passes within a single radius of the point
    http://codereview.stackexchange.com/questions/86421/line-segment-to-circle-collision-algorithm

    Args:
        point(np.ndarray): size 2 numpy array
        radius(float): radius of the circle
        trajectory: Nx2 matrix of (x,y) trajectory waypoints
        t(float): the t value of the trajectory to start searching from
        wrap(bool): if True, wrap the trajectory around to the beginning if the end is reached

    Returns:
        projection(np.ndarray): size 2 numpy array of the nearest point on the trajectory
        dist(float): distance from the point to the projection
        t(float): the t value of the projection along the trajectory
        min_dist_segment(int): the index of the segment of the trajectory that the projection is on
    """
    start_i = int(t)
    start_t = t % 1.0
    first_t = None
    first_i = None
    first_p = None
    trajectory = np.ascontiguousarray(trajectory)
    for i in range(start_i, trajectory.shape[0] - 1):
        start = trajectory[i, :]
        end = trajectory[i + 1, :] + 1e-6
        V = np.ascontiguousarray(end - start)

        a = np.dot(V, V)
        b = 2.0 * np.dot(V, start - point)
        c = np.dot(start, start) + np.dot(point, point) - 2.0 * np.dot(start, point) - radius * radius
        discriminant = b * b - 4 * a * c

        if discriminant < 0:
            continue
        #   print "NO INTERSECTION"
        # else:
        # if discriminant >= 0.0:
        discriminant = np.sqrt(discriminant)
        t1 = (-b - discriminant) / (2.0 * a)
        t2 = (-b + discriminant) / (2.0 * a)
        if i == start_i:
            if t1 >= 0.0 and t1 <= 1.0 and t1 >= start_t:
                first_t = t1
                first_i = i
                first_p = start + t1 * V
                break
            if t2 >= 0.0 and t2 <= 1.0 and t2 >= start_t:
                first_t = t2
                first_i = i
                first_p = start + t2 * V
                break
        elif t1 >= 0.0 and t1 <= 1.0:
            first_t = t1
            first_i = i
            first_p = start + t1 * V
            break
        elif t2 >= 0.0 and t2 <= 1.0:
            first_t = t2
            first_i = i
            first_p = start + t2 * V
            break
    # wrap around to the beginning of the trajectory if no intersection is found1
    if wrap and first_p is None:
        for i in range(-1, start_i):
            start = trajectory[i % trajectory.shape[0], :]
            end = trajectory[(i + 1) % trajectory.shape[0], :] + 1e-6
            V = end - start

            a = np.dot(V, V)
            b = 2.0 * np.dot(V, start - point)
            c = np.dot(start, start) + np.dot(point, point) - 2.0 * np.dot(start, point) - radius * radius
            discriminant = b * b - 4 * a * c

            if discriminant < 0:
                continue
            discriminant = np.sqrt(discriminant)
            t1 = (-b - discriminant) / (2.0 * a)
            t2 = (-b + discriminant) / (2.0 * a)
            if t1 >= 0.0 and t1 <= 1.0:
                first_t = t1
                first_i = i
                first_p = start + t1 * V
                break
            elif t2 >= 0.0 and t2 <= 1.0:
                first_t = t2
                first_i = i
                first_p = start + t2 * V
                break

    return first_p, first_i, first_t


@njit(cache=True)
def get_actuation(pose_theta, lookahead_point, position, lookahead_distance, wheelbase):
    waypoint_y = np.dot(np.array([np.sin(-pose_theta), np.cos(-pose_theta)]), lookahead_point[0:2] - position)
    speed = lookahead_point[2]
    if np.abs(waypoint_y) < 1e-6:
        return speed, 0.
    radius = 1 / (2.0 * waypoint_y / lookahead_distance ** 2)
    steering_angle = np.arctan(wheelbase / radius)
    return speed, steering_angle


@njit(cache=True)
def get_current_waypoint(waypoints, max_reacquire, lookahead_distance, position, n_next_points=5):
    """Get the current waypoint.

    Args:
        waypoints(np.ndarray): the waypoints
        lookahead_distance(float): the lookahead distance
        position(np.ndarray): the current position
        theta(float): the current pose angle
        n_next_points(int): the number of next points to return

    Returns:
        current_waypoint(np.ndarray): the current waypoint
    """
    wpts = waypoints
    nearest_point, nearest_dist, t, i = nearest_point_on_trajectory(position, wpts[:, 0:2])

    if nearest_dist < lookahead_distance:
        lookahead_point, i2, t2 = first_point_on_trajectory_intersecting_circle(
            position, lookahead_distance, wpts[:, 0:2],
            i + t, wrap=True
        )
        if i2 is None:
            return None

        current_waypoint = np.empty((n_next_points, 4))

        if i2 + n_next_points > wpts.shape[0]:
            o = i2 + n_next_points - wpts.shape[0]
            current_waypoint[:n_next_points - o, :] = wpts[i2:, :]
            if o != 0:
                current_waypoint[n_next_points - o:, :] = wpts[:o, :]
        else:
            if i2 < 0:
                current_waypoint[0:(-i2), :] = wpts[i2 - 1:i2, :]
                current_waypoint[(-i2):, :] = wpts[0:i2 + n_next_points, :]

            else:
                current_waypoint[:, :] = wpts[i2:i2 + n_next_points, :]

        return current_waypoint

    elif nearest_dist < max_reacquire:
        current_waypoint = np.empty((n_next_points, 4))

        if i + n_next_points > wpts.shape[0]:
            o = i + n_next_points - wpts.shape[0]
            # x, y
            current_waypoint[:n_next_points - o, :] = wpts[i:, :]
            if o != 0:
                current_waypoint[n_next_points - o:, :] = wpts[:o, :]

        else:
            current_waypoint[:, :] = wpts[i:i + n_next_points, :]
        return current_waypoint
    else:
        return None


@njit(cache=True)
def pi_2_pi(angle):
    if angle > np.pi:
        return angle - 2.0 * np.pi
    if angle < -np.pi:
        return angle + 2.0 * np.pi
    return angle


@njit(cache=True)
def zero_2_2pi(angle):
    if angle > 2 * np.pi:
        return angle - 2.0 * np.pi
    if angle < 0:
        return angle + 2.0 * np.pi
    return angle


@njit(cache=True)
def pi_2_pi2(angle):
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    return np.arctan2(sin_angle, cos_angle)


@njit(cache=True)
def get_rotation_matrix(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.ascontiguousarray(np.array([[c, -s], [s, c]]))


@njit(cache=True)
def quat_2_rpy(x, y, z, w):
    """
    Converts a quaternion into euler angles (roll, pitch, yaw)

    Args:
        x, y, z, w (float): input quaternion

    Returns:
        r, p, y (float): roll, pitch yaw
    """
    t0 = 2. * (w * x + y * z)
    t1 = 1. - 2. * (x * x + y * y)
    roll = np.atan2(t0, t1)

    t2 = 2. * (w * y - z * x)
    t2 = 1. if t2 > 1. else t2
    t2 = -1. if t2 < -1. else t2
    pitch = np.asin(t2)

    t3 = 2. * (w * z + x * y)
    t4 = 1. - 2. * (y * y + z * z)
    yaw = np.atan2(t3, t4)
    return roll, pitch, yaw


@njit(cache=True)
def indexOfFurthestPoint(vertices, d):
    """Return the index of the vertex furthest away along a direction in the list of vertices.

    Args:
        vertices (np.ndarray, (n, 2)): the vertices we want to find avg on

    Returns:
        idx (int): index of the furthest point
    """
    return np.argmax(vertices.dot(d))


@njit(cache=True)
def support(vertices1, vertices2, d):
    """
    Minkowski sum support function for GJK
    Args:
        vertices1 (np.ndarray, (n, 2)): vertices of the first body
        vertices2 (np.ndarray, (n, 2)): vertices of the second body
        d (np.ndarray, (2, )): direction to find the support along
    Returns:
        support (np.ndarray, (n, 2)): Minkowski sum
    """
    i = indexOfFurthestPoint(vertices1, d)
    j = indexOfFurthestPoint(vertices2, -d)
    return vertices1[i] - vertices2[j]


@njit(cache=True)
def perpendicular(pt):
    """Return a 2-vector's perpendicular vector.

    Args:
        pt (np.ndarray, (2,)): input vector

    Returns:
        pt (np.ndarray, (2,)): perpendicular vector
    """
    temp = pt[0]
    pt[0] = pt[1]
    pt[1] = -1 * temp
    return pt


@njit(cache=True)
def tripleProduct(a, b, c):
    """Return triple product of three vectors.

    Args:
        a, b, c (np.ndarray, (2,)): input vectors

    Returns:
        (np.ndarray, (2,)): triple product
    """
    ac = a.dot(c)
    bc = b.dot(c)
    return b * ac - a * bc


@njit(cache=True)
def avgPoint(vertices):
    """Return the average point of multiple vertices.

    Args:
        vertices (np.ndarray, (n, 2)): the vertices we want to find avg on

    Returns:
        avg (np.ndarray, (2,)): average point of the vertices
    """
    return np.sum(vertices, axis=0) / vertices.shape[0]


@njit(cache=True)
def curvature(x, y, i):
    # Get points before and after the current point
    x_prev, y_prev = x[i - 1], y[i - 1]
    x_curr, y_curr = x[i], y[i]
    x_next, y_next = x[i + 1], y[i + 1]

    # Calculate the derivatives
    dxdt = (x_next - x_prev) / 2.0
    dydt = (y_next - y_prev) / 2.0
    dx2dt2 = x_next - 2 * x_curr + x_prev
    dy2dt2 = y_next - 2 * y_curr + y_prev

    # Calculate curvature using the formula
    k = (dxdt * dy2dt2 - dydt * dx2dt2) / ((dxdt ** 2 + dydt ** 2) ** (3 / 2))

    return k


def moving_average(data, window_size):
    cumsum = np.cumsum(data)
    cumsum[window_size:] = cumsum[window_size:] - cumsum[:-window_size]
    return cumsum[window_size - 1:] / window_size


@njit(cache=True)
def distance_between_points(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


@njit(cache=True)
def find_close_obstacles(current_pos, obstacles):
    # Calculate the distance to the closest obstacle for each of the corners
    t_vec_all_length_center = np.clip(np.sqrt(np.sum((current_pos - obstacles) ** 2, axis=1)), 0.01, 100)
    close_ob = obstacles[np.argsort(t_vec_all_length_center)[:50]]

    # t_vec_all_length = np.zeros((obstacles.shape[0], 6))
    # t_vec_all_length[:, 0] = np.sqrt(np.sum((current_pos - np.array([0.15, 0.25]) - obstacles) ** 2, axis=1))
    # t_vec_all_length[:, 1] = np.sqrt(np.sum((current_pos - np.array([-0.15, 0.25]) - obstacles) ** 2, axis=1))
    # t_vec_all_length[:, 2] = np.sqrt(np.sum((current_pos - np.array([0.15, -0.25]) - obstacles) ** 2, axis=1))
    # t_vec_all_length[:, 3] = np.sqrt(np.sum((current_pos - np.array([-0.15, -0.25]) - obstacles) ** 2, axis=1))
    # t_vec_all_length[:, 4] = np.sqrt(np.sum((current_pos - np.array([0.15, 0.0]) - obstacles) ** 2, axis=1))
    # t_vec_all_length[:, 5] = np.sqrt(np.sum((current_pos - np.array([-0.15, 0.0]) - obstacles) ** 2, axis=1))
    # t_vec_all_length = np.clip(t_vec_all_length, 0.01, 100)
    # Select the 50 closest obstacles
    # t_vec_all_length_idx = np.argmin(t_vec_all_length, axis=0)
    # close_ob = obstacles[t_vec_all_length_idx]
    return close_ob


@njit(cache=True)
def get_trmtx(pose):
    """Get transformation matrix of vehicle frame -> global frame,

    Args:
        pose (np.ndarray (3, )): current pose of the vehicle

    return:
        H (np.ndarray (4, 4)): transformation matrix
    """
    x = pose[0]
    y = pose[1]
    th = pose[2]
    cos = np.cos(th)
    sin = np.sin(th)
    H = np.array([[cos, -sin, 0., x], [sin, cos, 0., y], [0., 0., 1., 0.], [0., 0., 0., 1.]])
    return H


@njit(cache=True)
def get_vertices(pose, length, width):
    """Utility function to return vertices of the car body given pose and size.

    Args:
        pose (np.ndarray, (3, )): current world coordinate pose of the vehicle
        length (float): car length
        width (float): car width

    Returns:
        vertices (np.ndarray, (4, 2)): corner vertices of the vehicle body
    """
    H = get_trmtx(pose)
    rl = H.dot(np.asarray([[-length / 2], [width / 2], [0.], [1.]])).flatten()
    rr = H.dot(np.asarray([[-length / 2], [-width / 2], [0.], [1.]])).flatten()
    fl = H.dot(np.asarray([[length / 2], [width / 2], [0.], [1.]])).flatten()
    fr = H.dot(np.asarray([[length / 2], [-width / 2], [0.], [1.]])).flatten()
    rl = rl / rl[3]
    rr = rr / rr[3]
    fl = fl / fl[3]
    fr = fr / fr[3]
    vertices = np.asarray([[rl[0], rl[1]], [rr[0], rr[1]], [fr[0], fr[1]], [fl[0], fl[1]]])

    # c = np.cos(pose[2])
    # s = np.sin(pose[2])
    # x, y = pose[0], pose[1]
    # tl_x = -length / 2 * c + width / 2 * (-s) + x
    # tl_y = -length / 2 * s + width / 2 * c + y
    # tr_x = length / 2 * c + width / 2 * (-s) + x
    # tr_y = length / 2 * s + width / 2 * c + y
    # bl_x = -length / 2 * c + (-width / 2) * (-s) + x
    # bl_y = -length / 2 * s + (-width / 2) * c + y
    # br_x = length / 2 * c + (-width / 2) * (-s) + x
    # br_y = length / 2 * s + (-width / 2) * c + y
    # vertices = np.asarray([[tl_x, tl_y], [bl_x, bl_y], [br_x, br_y], [tr_x, tr_y]])
    #
    return vertices


@njit(cache=True)
def xy_2_rc(x, y, orig_x, orig_y, orig_c, orig_s, height, width, resolution):
    """Translate (x, y) coordinate into (r, c) in the matrix.

    Args:
        x (float): coordinate in x (m)
        y (float): coordinate in y (m)
        orig_x (float): x coordinate of the map origin (m)
        orig_y (float): y coordinate of the map origin (m)

    Returns:
        r (int): row number in the transform matrix of the given point
        c (int): column number in the transform matrix of the given point
    """
    # translation
    x_trans = x - orig_x
    y_trans = y - orig_y

    # rotation
    x_rot = x_trans * orig_c + y_trans * orig_s
    y_rot = -x_trans * orig_s + y_trans * orig_c

    # clip the state to be a cell
    if x_rot < 0 or x_rot >= width * resolution or y_rot < 0 or y_rot >= height * resolution:
        c = -1
        r = -1
    else:
        c = int(x_rot / resolution)
        r = int(y_rot / resolution)

    return r, c


@njit(cache=True)
def cross(v1, v2):
    """Cross product of two 2-vectors.

    Args:
        v1, v2 (np.ndarray(2, )): input vectors

    Returns:
        crossproduct (float): cross product
    """
    return v1[0] * v2[1] - v1[1] * v2[0]


@njit(cache=True)
def are_collinear(pt_a, pt_b, pt_c):
    """Checks if three points are collinear in 2D.

    Args:
        pt_a, pt_b, pt_c (np.ndarray(2, )): points to check in 2D

    Returns:
        col (bool): whether three points are collinear
    """
    tol = 1e-8
    ba = pt_b - pt_a
    ca = pt_a - pt_c
    col = np.fabs(cross(ba, ca)) < tol
    return col

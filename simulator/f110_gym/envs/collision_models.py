import numpy as np
from numba import njit

from f110_gym.envs.env_utils import avgPoint, tripleProduct, perpendicular, support


@njit(cache=True)
def collision(vertices1, vertices2):
    """GJK test to see whether two bodies overlap.

    Args:
        vertices1 (np.ndarray, (n, 2)): vertices of the first body
        vertices2 (np.ndarray, (n, 2)): vertices of the second body

    Returns:
        overlap (boolean): True if two bodies collide
    """
    index = 0
    simplex = np.empty((3, 2))

    position1 = avgPoint(vertices1)
    position2 = avgPoint(vertices2)

    d = position1 - position2

    if d[0] == 0 and d[1] == 0:
        d[0] = 1.0

    a = support(vertices1, vertices2, d)
    simplex[index, :] = a

    if d.dot(a) <= 0:
        return False

    d = -a

    iter_count = 0
    while iter_count < 1e3:
        a = support(vertices1, vertices2, d)
        index += 1
        simplex[index, :] = a
        if d.dot(a) <= 0:
            return False

        ao = -a

        if index < 2:
            b = simplex[0, :]
            ab = b - a
            d = tripleProduct(ab, ao, ab)
            if np.linalg.norm(d) < 1e-10:
                d = perpendicular(ab)
            continue

        b = simplex[1, :]
        c = simplex[0, :]
        ab = b - a
        ac = c - a

        acperp = tripleProduct(ab, ac, ac)

        if acperp.dot(ao) >= 0:
            d = acperp
        else:
            abperp = tripleProduct(ac, ab, ab)
            if abperp.dot(ao) < 0:
                return True
            simplex[0, :] = simplex[1, :]
            d = abperp

        simplex[1, :] = simplex[2, :]
        index -= 1

        iter_count += 1
    return False


@njit(cache=True)
def collision_multiple(vertices):
    """Check pair-wise collisions for all provided vertices.

    Args:
        vertices (np.ndarray (num_bodies, 4, 2)): all vertices for checking pair-wise collision

    Returns:
        collisions (np.ndarray (num_vertices, )): whether each body is in collision
        collision_idx (np.ndarray (num_vertices, )): which index of other body is each index's body is in collision,
            -1 if not in collision
    """
    collisions = np.zeros((vertices.shape[0],))
    collision_idx = -1 * np.ones((vertices.shape[0],))

    for i in range(vertices.shape[0] - 1):
        for j in range(i + 1, vertices.shape[0]):
            # check collision
            vi = np.ascontiguousarray(vertices[i, :, :])
            vj = np.ascontiguousarray(vertices[j, :, :])
            ij_collision = collision(vi, vj)
            # fill in results
            if ij_collision:
                collisions[i] = 1.
                collisions[j] = 1.
                collision_idx[i] = j
                collision_idx[j] = i

    return collisions, collision_idx

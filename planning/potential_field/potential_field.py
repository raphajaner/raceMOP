from functools import partial
import numpy as np
import jax.numpy as jnp
from jax import grad
from jax import jit as jax_jit
import time
from matplotlib import pyplot as plt, patches


@partial(jax_jit, static_argnames=['rr', 'k_att', 'k_rep'])
def make_potential_field(current_pos, goal, obstacles, rr, k_att, k_rep):
    t_vec_all_length_1 = jnp.clip(
        jnp.linalg.norm(current_pos - np.array([0.15, 0.25]) - obstacles, axis=1),
        0.01, 100)
    t_vec_all_length_2 = jnp.clip(
        jnp.linalg.norm(current_pos - np.array([-0.15, 0.25]) - obstacles, axis=1),
        0.01, 100)
    t_vec_all_length_3 = jnp.clip(
        jnp.linalg.norm(current_pos - np.array([0.15, -0.25]) - obstacles, axis=1),
        0.01, 100)
    t_vec_all_length_4 = jnp.clip(
        jnp.linalg.norm(current_pos - np.array([-0.15, -0.25]) - obstacles, axis=1),
        0.01, 100)
    t_vec_all_length_5 = jnp.clip(
        jnp.linalg.norm(current_pos - np.array([0.15, 0.0]) - obstacles, axis=1),
        0.01, 100)
    t_vec_all_length_6 = jnp.clip(
        jnp.linalg.norm(current_pos - np.array([-0.15, 0.0]) - obstacles, axis=1),
        0.01, 100)

    t_vec_all_length = jnp.concatenate(
        [
            t_vec_all_length_1[:, None],
            t_vec_all_length_2[:, None],
            t_vec_all_length_3[:, None],
            t_vec_all_length_4[:, None],
            t_vec_all_length_5[:, None],
            t_vec_all_length_6[:, None]
        ],
        axis=1)

    # Take the minimum of all distances for each of the corners (axis=0 means take the minimum of each column=corner)
    t_vec_all_length = jnp.min(t_vec_all_length, axis=0)
    rep = 0.5 * k_rep * (1 / t_vec_all_length)
    rep = jnp.sum(rep)

    # Attractive point (goal)
    goal_length = jnp.clip(jnp.linalg.norm(goal - current_pos), 0.01, 20)
    att = k_att * 0.5 * goal_length * 2
    att = jnp.sum(att)

    # Create the potential
    potential = rep + att

    return potential


@partial(jax_jit, static_argnames=['rr', 'k_att', 'k_rep'])
def grad_potential_field(current_pos, angle, goal, obstacles, rr, k_att, k_rep):
    return grad(make_potential_field)(current_pos, goal, obstacles, rr, k_att, k_rep)


# @partial(jax_jit, static_argnames=['rr', 'k_att', 'k_rep', 'step_size'])
def step_potential_field(current_pos, angle, goal, obstacles, rr, k_att, k_rep, step_size):
    # cannot be jited because of temporal array
    grad = grad_potential_field(current_pos, angle, goal, obstacles, rr, k_att, k_rep).block_until_ready()
    norm = np.linalg.norm(grad)
    grad_norm = grad / (norm + 1e-5)

    current_pos = current_pos - grad_norm * step_size
    return np.array(current_pos, dtype=np.float64), angle


def plot_potential_fields(figure, scans, close_obstacles, target_x, target_y, path_smooth, x_wp, y_wp):

    if time.time() % 0.1 > 0.01:
        ax = figure.gca()
        plt.cla()
        # turn grid on
        # ax.grid(True)
        ax.scatter(scans[:, 0], scans[:, 1], color="blue", s=0.5)
        # ax.scatter(close_obstacles[:, 0], close_obstacles[:, 1], color="orange", s=2.0)

        # plot car as a rectangle with size 0.3 x 0.5 and center at (0,0)
        ax.add_patch(patches.Rectangle((-0.15, -0.25), 0.3, 0.5, color="black"))

        # add a circle with radius 0.5 around the car
        # circle = patches.Circle((0, 0), 1.2, color="black", fill=False)
        # ax.add_patch(circle)
        # circle = patches.Circle((0, 0), 2.0, color="blue", fill=False)
        # ax.add_patch(circle)

        # set limits
        # ax.set_xlim(-4, 4)
        # ax.set_ylim(-4, 10.1)

        # Get active obstacles
        # pf_center_init = obstacles[t_vec_true_init]
        # plt.scatter(pf_center_init[:, 0], pf_center_init[:, 1], color='yellow')

        # target point

        # ego position
        ax.scatter(0, 0, color="purple", s=0.5)

        # path
        # ax.plot(path[:, 0], path[:, 1], color="black", marker='.', markersize=0.5)

        # path
        # try:
        #     ax.plot(path_smooth[:, 0], path_smooth[:, 1], color="yellow", marker='*', markersize=0.5)
        # except:
        #     pass
        # inter goal
        # plt.scatter(inter_goal[0], inter_goal[1], color="orange")

        # gaps
        # for i in range(len(gaps_idx)):
        #     plt.plot([x_map_scans[gaps_idx[i][0]], x_map_scans[gaps_idx[i][1]]],
        #              [y_map_scans[gaps_idx[i][0]], y_map_scans[gaps_idx[i][1]]], color="red")

        # target waypoint
        # ax.scatter(x_wp, y_wp, color="orange")

        # ax.scatter(target_x, target_y, color="green")
        # add limits
        plt.pause(0.000000001)

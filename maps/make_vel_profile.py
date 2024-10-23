import trajectory_planning_helpers as optim
import numpy as np
import hydra
from hydra.core.hydra_config import DictConfig


@hydra.main(version_base=None, config_path='../configs/', config_name='config')
def main(config: DictConfig) -> None:
    """Main function to run the training.

    Args:
        config (DictConfig): Configuration object.
    """
    optim_paths = dict()

    map_names = config.maps.maps_train + config.maps.maps_test
    for map_name in map_names:
        print(f'Working on map: {map_name}')
        map_path = config.maps.map_path + f'{map_name}/{map_name}_map'
        # wpt_path = config.maps.map_path + f'{map_name}/{map_name}_raceline.csv'
        wpt_path = config.maps.map_path + f'{map_name}/{map_name}_centerline.csv'
        # wpt_path = config.maps.map_path + f'{map_name}/{map_name}_centerline_vel_newconv.csv'
        # waypoints = np.loadtxt(wpt_path, delimiter=';', skiprows=3)
        waypoints = np.loadtxt(wpt_path, delimiter=',', skiprows=1)

        el_lengths = np.linalg.norm(np.diff(waypoints[:, :2], axis=0), axis=1)
        psi, kappa = optim.calc_head_curv_num.calc_head_curv_num(waypoints[:, :2], el_lengths, False)
        nvecs = optim.calc_normal_vectors.calc_normal_vectors(psi)
        crossing = optim.check_normals_crossing.check_normals_crossing(waypoints, nvecs)
        if crossing:
            print(f"Major problem: nvecs are crossing. Result will be incorrect. Fix the center line file.")
        else:
            print(f"nvecs are not crossing. Everything is fine.")

        # el_lengths = np.linalg.norm(waypoints[3, [0, 1]] - waypoints[4, [0, 1]])
        # el_lengths = np.repeat(el_lengths, path.shape[0])
        # el_lengths = np.cumsum(el_lengths)
        # el_lengths = np.cumsum(el_lengths)
        # [x, y, w_tr_right, w_tr_left, (banking)]

        # distance between waypoints in path [[x,y],..]
        import trajectory_planning_helpers.calc_spline_lengths
        # path_smoothed, w_tr_right_smoothed_cl[:-1], w_tr_left_smoothed_cl[:-1]
        track_reg = optim.spline_approximation.spline_approximation(waypoints, stepsize_prep=0.1, stepsize_reg=0.2,
                                                                    debug=True)

        #
        # psi, kappa = optim.calc_head_curv_num.calc_head_curv_num(track_reg[:, :2], el_lengths, False)

        el_lengths = np.linalg.norm(np.diff(track_reg[:, :2], axis=0), axis=1)
        el_lengths = np.insert(el_lengths, -1, el_lengths[-1])

        psi, kappa = optim.calc_head_curv_num.calc_head_curv_num(track_reg[:, :2], el_lengths, True)

        nvecs = optim.calc_normal_vectors.calc_normal_vectors(psi)
        crossing = optim.check_normals_crossing.check_normals_crossing(track_reg, nvecs)
        if crossing:
            print(f"Major problem: nvecs are crossing. Result will be incorrect. Fix the center line file.")
        else:
            print(f"nvecs are not crossing. Everything is fine.")

        path = np.vstack((track_reg[:, 0], track_reg[:, 1])).T

        # import pdb
        # pdb.set_trace()
        # el_lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
        # el_lengths = np.insert(el_lengths, -1, 0.2)
        #
        # # el_lengths = np.repeat(el_lengths, path.shape[0])
        #
        # # waypoints[:, [0, 1]]
        # # el_lengths = waypoints[:, 0]
        #
        # # kappa = waypoints[:, 4]
        #
        # psi, kappa = optim.calc_head_curv_num.calc_head_curv_num(
        #     path, el_lengths, is_closed=True
        # )

        ax_max_machines = np.array([[0.0, 5.0], [8.0, 7.51]])
        # ax_max_machines = np.array([[0.0, 4], [5.0, 4], [8.0, 4]])

        # ggv = np.array([[0, 0.0001, 0.0001], [4, 0.0001, 0.0001], [8, 0.0001, 0.0001]])
        ggv = np.array([[0, 4.0, 4.0], [8, 7.51, 3.0]])

        vmax = optim.calc_vel_profile.calc_vel_profile(
            ax_max_machines=ax_max_machines,
            kappa=kappa,  # curvature
            el_lengths=np.cumsum(el_lengths),
            closed=True,
            m_veh=3.47,
            drag_coeff=0,
            ggv=ggv,
            mu=np.repeat([0.80], kappa.shape),
            # v_start=0.0
        )

        # new conv
        heading_np = psi
        heading_np += np.pi / 2
        heading_np[heading_np > 2 * np.pi] -= 2 * np.pi
        heading_np[heading_np < 0] += 2 * np.pi

        # s_m	 x_m	 y_m	 psi_rad	 kappa_radpm	 vx_mps	 ax_mps2
        out = np.array([el_lengths, path[:, 0], path[:, 1], heading_np, kappa, vmax]).T
        wpt_path_new = config.maps.map_path + f'{map_name}/{map_name}_centerline_vel_newconv.csv'
        header = '\n\ns_m; x_m; y_m; psi_rad; kappa_radpm; vx_mps'  # ; ax_mps2'
        #
        np.savetxt(wpt_path_new, out, header=header, delimiter=";")
        print(f'Saved new centerline to: {wpt_path_new}')

        # plt.figure(3)
        # plt.clf()
        # plt.plot(self.track[:, 0], self.track[:, 1], '-', linewidth=2, color='blue', label="Track")
        # plt.plot(self.min_curve_path[:, 0], self.min_curve_path[:, 1], '-', linewidth=2, color='red', label="Raceline")
        #
        # l_line = self.track[:, 0:2] + self.nvecs * self.track[:, 2][:, None]
        # r_line = self.track[:, 0:2] - self.nvecs * self.track[:, 3][:, None]
        # plt.plot(l_line[:, 0], l_line[:, 1], linewidth=1, color='green', label="Boundaries")
        # plt.plot(r_line[:, 0], r_line[:, 1], linewidth=1, color='green')
        #
        # plt.legend()
        # plt.gca().set_aspect('equal', adjustable='box')
        # plt.xticks([])
        # plt.yticks([])
        # plt.tight_layout()
        # plt.savefig(f"racelines/minimum_curvature_path_{self.map_name}.svg", pad_inches=0)


if __name__ == "__main__":
    main()

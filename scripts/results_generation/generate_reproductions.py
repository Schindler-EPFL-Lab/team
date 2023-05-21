import math
from pathlib import Path

import numpy as np

from team.aligned_trajectories import AlignedTrajectories
from team.dynamical_movement_primitives import DynamicMovementPrimitives
from team.gaussian_mixture_regression import GMR
from team.probabilistic_encoding import ProbabilisticEncoding


def apply_noise(array: np.ndarray, noise: np.ndarray) -> np.ndarray:
    """
    Get noise vector and sum to unit vector to form noise matrix.
    Multiply array with noise matrix to apply noise to each entry.

    :param array: initial array
    :param noise: noise vector
    :return: noisy array
    """
    assert np.shape(array) == np.shape(noise), "dimensions are not correct"
    nb_dims = np.shape(noise)[0]
    noise = noise + np.ones(nb_dims)
    noise_matrix = np.diag(noise)
    return np.matmul(array, noise_matrix)


def main():
    task_list = [
        "opening",
        "door_closing",
        "brush_homing",
        "brush_picking_up",
        "rail_cleaning",
    ]
    base_folder = Path(__file__).parent.parent.parent.joinpath("data/maintenance_tasks")
    base_folder.mkdir()
    for task in task_list:
        j = 0
        nb_transf = 6  # ["tr", "rot", "scal", "ns_init", "ns_target", "ns_both"]
        nb_repro = 30
        # data loading
        data_dir = Path(__file__).parent.parent.parent.joinpath(
            f"data/learning_from_demonstrations/{task}"
        )
        # create task folder
        folder_path = base_folder.joinpath(f"{task}")
        folder_path.mkdir()
        # data preprocessing
        trajectories = AlignedTrajectories.load_dataset_and_preprocess(str(data_dir))
        # GMM
        pe = ProbabilisticEncoding(
            trajectories, max_nb_components=10, min_nb_components=4, iterations=20
        )
        # GMR
        regression = GMR(trajectories, pe)
        target = regression.prediction[-1, 1:]
        initial_state = regression.prediction[0, 1:]
        dmp = DynamicMovementPrimitives(
            regression=regression.prediction,
            c_order=1,
            initial_joints=initial_state,
            goal_joints=target,
        )

        dmp.save_dmp(folder_path)

        # choose transformation to apply
        for i in range(1, nb_repro + 1):
            # DMP
            target = regression.prediction[-1, 1:]
            initial_state = regression.prediction[0, 1:]

            # counter to switch transformation type
            if i % nb_transf == 0:
                j = j + 1

            if j == 0:
                # apply translation on both initial and target references
                magnitude = np.random.uniform(0, 0.5)
                translation_matrix = np.array(
                    [
                        [1, 0, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 1, 0],
                        [magnitude, magnitude, magnitude, magnitude, magnitude, 1],
                    ]
                )
                target = np.matmul(target, translation_matrix)
                initial_state = np.matmul(initial_state, translation_matrix)
            elif j == 1:
                # apply rotation on both initial and target references
                angle_rad = np.random.uniform(-math.pi, math.pi)
                rotation_matrix = np.array(
                    [
                        [np.cos(angle_rad), np.sin(angle_rad), 0, 0, 0, 0],
                        [-np.sin(angle_rad), np.cos(angle_rad), 0, 0, 0, 0],
                        [0, 0, 1, 0, 0, 0],
                        [0, 0, 0, np.cos(angle_rad), np.sin(angle_rad), 0],
                        [0, 0, 0, -np.sin(angle_rad), np.cos(angle_rad), 0],
                        [0, 0, 0, 0, 0, 1],
                    ]
                )
                target = np.matmul(target, rotation_matrix)
                initial_state = np.matmul(initial_state, rotation_matrix)
            elif j == 2:
                # apply scaling on both initial and target references
                scaling_factor = np.random.uniform(0, 2)
                scaling_matrix = np.array(
                    [
                        [scaling_factor, 0, 0, 0, 0, 0],
                        [0, scaling_factor, 0, 0, 0, 0],
                        [0, 0, scaling_factor, 0, 0, 0],
                        [0, 0, 0, scaling_factor, 0, 0],
                        [0, 0, 0, 0, scaling_factor, 0],
                        [0, 0, 0, 0, 0, scaling_factor],
                    ]
                )
                target = np.matmul(target, scaling_matrix)
                initial_state = np.matmul(initial_state, scaling_matrix)
            elif j == 3:
                # apply noise on initial reference (simulate fixed target)
                noise_i = 0.2 * np.random.rand(6)
                target = regression.prediction[-1, 1:]
                initial_state = apply_noise(target, noise_i)
            elif j == 4:
                # apply noise on target reference (simulate fixed initial position)
                noise_t = 0.2 * np.random.rand(6)
                target = apply_noise(target, noise_t)
                initial_state = regression.prediction[0, 1:]
            elif j == 5:
                # apply noise on both initial and target references
                noise_b = 0.2 * np.random.rand(6)
                target = apply_noise(target, noise_b)
                initial_state = apply_noise(initial_state, noise_b)

            dmp_traj = dmp.compute_joint_dynamics(goal=target, y_init=initial_state)
            reproduction_folder = Path(folder_path, "reproduction" + str(i))
            reproduction_folder.mkdir()

            dmp_traj.save(
                path_to_file=Path(reproduction_folder, "task_config.json"),
                initial_state=initial_state,
                target=target,
            )


if __name__ == "__main__":
    main()

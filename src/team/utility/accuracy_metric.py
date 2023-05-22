import json
from pathlib import Path

import numpy as np
from rws2.RWS2 import RWS

from team.utility.lie_algebra_and_tf import quaternion_matrix, se3_inverse
from team.utility.optimizer import Optimizer


def endpoint_accuracy(rws: RWS) -> float:
    """
    Computes the homogeneous rotation matrix from the quaternions list and then add
    the translation vector to obtain a homogeneous transformation matrix
    appertaining in the Lie group SE(3).
    It does that to the target and robot endpoint pose transformations.
    Finally, it calculates the relative transformation and assesses the discrepancy.

    :param rws: RWS object to interface with the robot controller
    :return: the error between the relative transformation and the identity
    """

    # reads the endpoint pose computed through forward kinematics in RAPID
    endpoint_pose_lists = rws.get_robtarget_variables("endpoint_pose")
    endpoint_pose = endpoint_pose_lists[0] + endpoint_pose_lists[1]

    # reads the target pose in RAPID
    target_pose_lists = rws.get_robtarget_variables("target_pose")
    target_pose = target_pose_lists[0] + target_pose_lists[1]

    # get quaternions vectors as [qw, qx, qy, qz]
    target_ori = [target_pose[6]] + target_pose[3:6]
    endpoint_ori = [endpoint_pose[6]] + endpoint_pose[3:6]

    # get translation vectors in m instead of mm
    target_tran = [t / 1000 for t in target_pose[:3]]
    endpoint_tran = [c / 1000 for c in endpoint_pose[:3]]

    # SE3 matrices construction
    t_start = quaternion_matrix(endpoint_ori)
    t_start[:3, 3] = endpoint_tran
    t_goal = quaternion_matrix(target_ori)
    t_goal[:3, 3] = target_tran

    # relative transformation between target and current pose
    relative_t = np.dot(se3_inverse(t_start), t_goal)

    error = np.linalg.norm(relative_t - np.eye(4))

    return error


def endpoint_position_accuracy(rws: RWS) -> float:
    """
    Adds the translation vector to obtain a homogeneous transformation matrix
    appertaining in the Lie group SE(3).
    It does that to the target and robot endpoint pose transformations.
    Finally, it calculates the relative transformation and assesses the discrepancy.

    :param rws: RWS object to interface with the robot controller
    :return: the position error between the relative transformation and the identity
    """

    # reads the endpoint pose computed through forward kinematics in RAPID
    endpoint_pose_lists = rws.get_robtarget_variables("endpoint_pose")
    endpoint_pose = endpoint_pose_lists[0] + endpoint_pose_lists[1]

    # reads the target pose in RAPID
    target_pose_lists = rws.get_robtarget_variables("target_pose")
    target_pose = target_pose_lists[0] + target_pose_lists[1]

    # get translation vectors in m instead of mm
    target_tran = [t for t in target_pose[:3]]
    endpoint_tran = [c for c in endpoint_pose[:3]]

    # SE3 matrices construction
    t_start = np.eye(4)
    t_start[:3, 3] = endpoint_tran
    t_goal = np.eye(4)
    t_goal[:3, 3] = target_tran

    # relative transformation between target and current pose
    relative_t = np.dot(se3_inverse(t_start), t_goal)

    error = np.linalg.norm(relative_t - np.eye(4))

    return error


def endpoint_orientation_accuracy(rws: RWS) -> float:
    """
    Computes the homogeneous rotation matrix from the quaternions list to obtain a
    homogeneous transformation matrix appertaining in the Lie group SE(3).
    It does that to the target and robot endpoint pose transformations.
    Finally, it calculates the relative transformation and assesses the discrepancy.

    :param rws: RWS object to interface with the robot controller
    :return: the orientation error between the relative transformation and the identity
    """

    # reads the endpoint pose computed through forward kinematics in RAPID
    endpoint_pose_lists = rws.get_robtarget_variables("endpoint_pose")
    endpoint_pose = endpoint_pose_lists[0] + endpoint_pose_lists[1]

    # reads the target pose in RAPID
    target_pose_lists = rws.get_robtarget_variables("target_pose")
    target_pose = target_pose_lists[0] + target_pose_lists[1]

    # get quaternions vectors as [qw, qx, qy, qz]
    target_ori = [target_pose[6]] + target_pose[3:6]
    endpoint_ori = [endpoint_pose[6]] + endpoint_pose[3:6]

    # SE3 matrices construction
    t_start = quaternion_matrix(endpoint_ori)
    t_goal = quaternion_matrix(target_ori)

    # relative transformation between target and current pose
    relative_t = np.dot(se3_inverse(t_start), t_goal)

    error = np.linalg.norm(relative_t - np.eye(4))

    return error


def endpoint_joint_accuracy(rws: RWS, goal_j: np.ndarray) -> float:
    """
    Computes the endpoint joint error between the goal joints refence and the robot
    joints at the end of the trajectory.

    :param rws: RWS object to interface with the robot controller
    :param goal_j: goal joints reference
    :return: the joint error
    """

    return np.linalg.norm(goal_j - np.array(rws.get_joints_positions()))


def gmcc_similarity_metric(
    reproduced_trajectory: np.ndarray, learned_trajectory: np.ndarray
) -> float:
    """
    Computes the GMCC similarity metric between prediction and ground truth trajectory.
    GMCC is a measure of how well a linear transformation between 2 trajectories
    can be mapped.
    This similarity metric is invariant to and only to linear transformations.
    Paper arxiv url: https://arxiv.org/abs/1906.09802

    :param reproduced_trajectory: trajectory based on task goal requirements
    :param learned_trajectory: trajectory learned from the demonstrations dataset
    :return: the GMCC similary metric value between prediction and ground truth
    trajectory
    """

    regression = learned_trajectory
    reproduction = reproduced_trajectory
    matrix_h = Optimizer(regression, reproduction).find_optimum()
    prediction = np.matmul(reproduction, matrix_h)
    y_overbar = np.mean(regression, axis=0)
    num = np.linalg.norm(prediction - y_overbar)
    denom = np.linalg.norm(regression - y_overbar)
    assert denom != 0
    gmcc = num / denom
    return gmcc


def symmetric_gmcc(
    reproduced_trajectory: np.ndarray, learned_trajectory: np.ndarray
) -> float:
    """
    Computes the symmetric GMCC similarity metric between prediction and
    ground truth trajectory.

    :param reproduced_trajectory: trajectory based on task goal requirements
    :param learned_trajectory: trajectory learned from the demonstrations dataset
    :return: the symmetric GMCC similary metric
    """
    return 0.5 * (
        gmcc_similarity_metric(reproduced_trajectory, learned_trajectory)
        + gmcc_similarity_metric(learned_trajectory, reproduced_trajectory)
    )


def save_model_performance(
    dir_path: Path,
    rws: RWS,
    goal_joints: np.ndarray,
    exist_ok: bool = False,
) -> None:
    """
    Saves model performance for trajectory reproduction j.

    :param dir_path: directory path to store data
    :param rws:  RWS object to interface with the robot controller
    :param goal_joints: robot goal target joints
    :param exist_ok: boolean to allow file override
    """

    # create the parent folder if it does not exist
    if not dir_path.exists():
        try:
            dir_path.mkdir()
        except FileNotFoundError:
            raise FileNotFoundError(
                "Parent folder does not exists, please check the "
                "consistency of the provided path!"
            )
    # save model performance info to file
    model_perf_path = dir_path.joinpath("model_performance.json")
    if model_perf_path.exists() and not exist_ok:
        raise FileExistsError("Not allowed to override an existing file!")
    data = {
        "pos_error": endpoint_position_accuracy(rws),
        "ori_error": endpoint_orientation_accuracy(rws),
        "j_error": endpoint_joint_accuracy(rws, goal_joints),
    }
    with open(model_perf_path, "w") as f:
        json.dump(data, f)

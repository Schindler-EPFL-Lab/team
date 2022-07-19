import json
import numpy as np
from pathlib import Path

from rws2.RWS2 import RWS
from learning_from_demo.utility.lie_algebra_and_tf import quaternion_matrix, se3_inverse


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
    model_perf_path = dir_path.joinpath(
        "model_performance.json"
    )
    if model_perf_path.exists() and not exist_ok:
        raise FileExistsError("Not allowed to override an existing file!")
    data = {
        "pos_error": endpoint_position_accuracy(rws),
        "ori_error": endpoint_orientation_accuracy(rws),
        "j_error": endpoint_joint_accuracy(rws, goal_joints),
    }
    with open(model_perf_path, "w") as f:
        json.dump(data, f)

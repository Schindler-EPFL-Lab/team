import numpy as np

from rws2.RWS2 import RWS
from learning_from_demo.utility.lie_algebra_and_tf import quaternion_matrix, se3_inverse


def endpoint_accuracy(rws: RWS) -> float:
    """
    Computes the homogeneous rotation matrix from the quaternions list and then add
    the translation vector to obtain a homogeneous transformation matrix
    appertaining to the Lie group SE(3).
    It does that both for the target and robot endpoint pose transformations.
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

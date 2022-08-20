import numpy as np
import math

# epsilon for testing whether a number is close to zero
_EPS = np.finfo(float).eps * 4.0


def quaternion_matrix(quaternion: list):
    """
    Return homogeneous rotation matrix from quaternion.
    """
    q = np.array(quaternion[:4], dtype=np.float64, copy=True)
    nq = np.dot(q, q)
    if nq < _EPS:
        return np.identity(4)
    q *= math.sqrt(2.0 / nq)
    q = np.outer(q, q)
    return np.array(
        (
            (1.0 - q[1, 1] - q[2, 2], q[0, 1] - q[2, 3], q[0, 2] + q[1, 3], 0.0),
            (q[0, 1] + q[2, 3], 1.0 - q[0, 0] - q[2, 2], q[1, 2] - q[0, 3], 0.0),
            (q[0, 2] - q[1, 3], q[1, 2] + q[0, 3], 1.0 - q[0, 0] - q[1, 1], 0.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )


def se3_inverse(p: np.ndarray) -> np.ndarray:
    """
    :param p: absolute SE(3) pose
    :return: the inverted pose
    """
    r_inv = p[:3, :3].transpose()
    t_inv = -r_inv.dot(p[:3, 3])
    return se3(r_inv, t_inv)


def se3(r: np.ndarray = np.eye(3), t: np.ndarray = np.array([0, 0, 0])) -> np.ndarray:
    """
    :param r: SO(3) rotation matrix
    :param t: 3x1 translation vector
    :return: SE(3) transformation matrix
    """
    se3 = np.eye(4)
    se3[:3, :3] = r
    se3[:3, 3] = t
    return se3

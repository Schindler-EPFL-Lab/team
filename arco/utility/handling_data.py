from typing import Optional


def create_default_dict(keys: Optional[list[str]] = None):
    if not keys:
        keys = [
            "timestamp",
            "tcp_x",
            "tcp_y",
            "tcp_z",
            "tcp_q1",
            "tcp_q2",
            "tcp_q3",
            "tcp_q4",
            "cf1",
            "cf4",
            "cf6",
            "cfx",
            "joint_1",
            "joint_2",
            "joint_3",
            "joint_4",
            "joint_5",
            "joint_6",
        ]
    return {key: [] for key in keys}
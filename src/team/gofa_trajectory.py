import json
from pathlib import Path

import numpy as np
from spatialmath import SE3

from team.dmp_trajectory import DmpTrajectory


class GoFaTrajectory(DmpTrajectory):
    def __init__(self, trajectory: np.ndarray):
        super().__init__(trajectory)
        self._translations = [
            SE3(0, 0, 0),
            SE3(0, 0, 0),
            SE3(0, 0, 0),
            SE3(0, 0, 0),
            SE3(0, 0, 0),
            SE3(0, 0, 0),
        ]
        self.tcp = self._compute_tcp()

    def _compute_tcp(self) -> np.ndarray:
        # raise RuntimeError("Must be implemented")
        tcps = []
        for measure in range(len(self)):
            joints = self.get_joints_at_index(measure)
            assert len(joints) == len(self._translations)

            tcp = SE3(0, 0, 0)
            for i in range(len(joints)):
                if i in [0, 1, 2, 3, 4, 5, 6]:
                    tcp *= SE3.Rz(joints[0]) * self._translations[i]
            tcps.append(tcp.t)
        return np.array(tcps)

    @classmethod
    def from_config_file(cls, file: Path) -> "GoFaTrajectory":
        with open(file, "rb") as f:
            json_data = json.load(f)
            return cls(
                np.array(json_data["trajectory"]),
            )

import json
from pathlib import Path

import numpy as np

from team.trajectory import TrajectoryBase


class DmpTrajectory(TrajectoryBase):
    """
    Provides the dmp joint angle trajectory to track
    """

    def __init__(self, trajectory: np.ndarray):
        assert np.shape(trajectory)[1] == 6, "array dimensions are not correct"
        super().__init__(trajectory)

    @property
    def joints(self) -> np.ndarray:
        return self._trajectory

    def save(
        self, path_to_file: Path, initial_state: np.ndarray, target: np.ndarray
    ) -> None:
        outdict = {
            "starting_j": initial_state.tolist(),
            "goal_j": target.tolist(),
            "trajectory": self.joints.tolist(),
        }
        with open(path_to_file, "w") as f:
            json.dump(outdict, f)

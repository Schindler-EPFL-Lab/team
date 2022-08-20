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

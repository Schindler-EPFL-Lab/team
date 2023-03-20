from typing import Optional

import numpy as np


class JSComponent:
    def __init__(self, component: int, values: Optional[list[float]] = None) -> None:
        self.component_nb = component
        if values is None:
            values = []
        self.values = values
        self._mean: Optional[float] = None
        self._std: Optional[float] = None

    def append(self, value: float):
        self._mean = None
        self._std = None
        self.values.append(value)

    @property
    def mean(self) -> float:
        if self._mean is None:
            self._mean = np.mean(self.values)
        return self._mean

    @property
    def std(self) -> float:
        if self._std is None:
            self._std = np.std(self.values)
        return self._std

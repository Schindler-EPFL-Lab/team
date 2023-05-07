import numpy as np
from scipy.optimize import least_squares


class Optimizer:
    """
    Least squared error optimization to find linear transformation A between
    ground truth (regression trajectory) and reproduction trajectory.
    Prediction is given applying matrix A to reproduction trajectory.
    """

    def __init__(self, truth: np.ndarray, reproduction: np.ndarray) -> None:
        assert np.shape(truth) == np.shape(reproduction)
        self.truth = truth
        self.reproduction = reproduction
        self.n_dimension = np.shape(truth)[1]
        self.initial_matrix = np.random.random((self.n_dimension, self.n_dimension))

    def _to_vector(self, x: np.ndarray) -> np.ndarray:
        """
        Flatten the input matrix x in a vector.

        :param x: input matrix
        :return: flattened matrix
        """
        return x.flatten()

    def _to_matrix(self, x: np.ndarray) -> np.ndarray:
        """
        Restore matrix from flattened vector.

        :param x: input vector
        :return: restored matrix
        """
        return x.reshape(self.n_dimension, self.n_dimension)

    def _objective_function(self, transf: np.ndarray) -> float:
        """
        Objective function of optimization problem

        :param transf: transformation matrix to optimize
        :return: Euclidean norm of error between prediction and ground truth
        """
        transf_as_matrix = self._to_matrix(transf)
        self.prediction = np.matmul(self.reproduction, transf_as_matrix)
        return np.linalg.norm(self.prediction - self.truth)

    def find_optimum(self) -> np.ndarray:
        """
        Optimization routine

        :return: optimum matrix
        """
        result = least_squares(self._objective_function,
                               self._to_vector(self.initial_matrix))
        result.x = self._to_matrix(result.x)
        return result.x

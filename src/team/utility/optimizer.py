import numpy as np
from scipy.optimize import minimize


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
        self.initial_matrix = self._initial_matrix()

    def _initial_matrix(self) -> np.ndarray:
        """
        Random initialization of transformation matrix

        :return: random initial matrix
        """
        return np.random.random((self.n_dimension, self.n_dimension))

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
        self.prediction = np.matmul(self.reproduction, transf)
        return np.linalg.norm(self.prediction - self.truth)

    def find_optimum(self) -> np.ndarray:
        """
        Optimization routine

        :return: optimum matrix
        """
        def f(x) -> float:
            """
            Function to optimize

            :param x: variable in form of vector
            :return: objective function value
            """
            matrix = self._to_matrix(x)
            return self._objective_function(matrix)
        x_0 = self.initial_matrix
        result = minimize(f, self._to_vector(x_0), tol=1)
        result.x = self._to_matrix(result.x)
        print(result.success)
        print(result.message)
        print(result.nit)
        return result.x

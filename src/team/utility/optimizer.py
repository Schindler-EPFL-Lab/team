import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Least squared error optimization to find linear transformation A between
    ground truth (regression trajectory) and reproduction trajectory.
    Prediction is given applying matrix A to reproduction trajectory.
    """

    def __init__(self, truth: np.ndarray, reproduction: np.ndarray) -> None:
        assert np.shape(truth) == np.shape(reproduction)
        self.truth = truth
        self.reproduction = reproduction
        self.n_dimension = np.shape(truth)[1]
        self.initial_A = self._initial_matrix()

    def _initial_matrix(self) -> np.ndarray:
        """Random initialization of transformation matrix

        Returns:
            np.ndarray: random initial matrix
        """
        return np.random.random((self.n_dimension, self.n_dimension))

    def _to_vector(self, x: np.ndarray) -> np.ndarray:
        """Flatten the input matrix x in a vector.

        Args:
            x (np.ndarray): input matrix

        Returns:
            np.ndarray: flattened matrix
        """
        return x.flatten()

    def _to_matrix(self, x: np.ndarray) -> np.ndarray:
        """Restore matrix from flattened vector.

        Args:
            x (np.ndarray): input vector

        Returns:
            np.ndarray: restored matrix 
        """
        return x.reshape(self.n_dimension, self.n_dimension)

    def _func(self, transf: np.ndarray) -> float:
        """Objective function of optimization problem

        Args:
            transf (np.ndarray): transformation matrix to optimize

        Returns:
            float: Euclidean norm of error between prediction and ground truth
        """
        self.prediction = np.matmul(self.reproduction, transf)
        return np.linalg.norm(self.prediction - self.truth)

    def find_optimum(self) -> np.ndarray:
        """Optimization routine

        Returns:
            np.ndarray: optimum matrix
        """
        def f(x) -> float:
            """Function to optimize

            Args:
                x (_type_): variable in form of vector

            Returns:
                float: objective function value
            """
            matrix_A = self._to_matrix(x)
            return self._func(matrix_A)
        x_0 = self.initial_A
        result = minimize(f, self._to_vector(x_0), tol=1)
        result.x = self._to_matrix(result.x)
        print(result.success)
        print(result.message)
        print(result.nit)
        return result.x

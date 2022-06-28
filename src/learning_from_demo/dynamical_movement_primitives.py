import json
from pathlib import Path
from typing import Optional

import numpy as np
from skopt import gp_minimize

from learning_from_demo.dmp_trajectory import DmpTrajectory


class DynamicMovementPrimitives:
    """
    Dynamic movement primitives implementation, implemented with the work and code of
    A. J. Ijspeert, J. Nakanishi, H. Hoffmann, P. Pastor, and S. Schaal, "Dynamical
    movement primitives: Learning attractor models for motor behaviors," Neural Com-
    putationvol. 25, no. 2, pp. 328–373, 2013. doi: 10.1162/NECO_a_00393.
    The class receives the general behavior of the desired motion as well as the task
    parameters (e.g goal, number of basis functions, order of dynamical system, etc.)
    and it computes the resulting joint angles dynamics

    ```python
    pe = ProbabilisticEncoding(
        trajectories,
        max_nb_components=10,
        min_nb_components=2,
        iterations=1,
        random_state=0,
    )
    regression = GMR(trajectories, pe)
    dmp = DynamicMovementPrimitives(
        regression_fct=regression,
        c_order=1,
        goal_joints=goal_joints,
        initial_joints=initial_joints,
    )
    ```

    :param regression: regression function extracted from dataset
    :param c_order: order of the dynamical system
    :param goal_joints: goal robot joints
    :param initial_joints: initial robot joints
    """

    _alpha_z: np.ndarray
    _beta_z: np.ndarray
    _alpha_g: np.ndarray
    _alpha_x: np.ndarray
    _alpha_v: np.ndarray
    _beta_v: np.ndarray
    _n_rfs: int
    _c: np.ndarray
    _c_d: np.ndarray
    _D: np.ndarray
    psi_history: np.ndarray
    w_history: np.ndarray

    def __init__(
        self,
        regression: np.ndarray,
        c_order: int,
        goal_joints: np.ndarray,
        initial_joints: np.ndarray,
        search_space: Optional[list[tuple[int, int]]] = None,
        dmp_parameters: Optional[tuple[np.ndarray, int]] = None,
    ) -> None:

        if search_space is None:
            search_space = [(5, 30), (50, 200)]
        self.sampling_rate = 1 / (regression[1, 0] - regression[0, 0])
        self.dt = regression[1, 0] - regression[0, 0]
        self.regression = regression
        # extract information
        self.timestamp = regression[:, 0]
        self._y_demo = regression[:, 1:]
        self._yd_demo, self._ydd_demo = self._compute_derivatives()
        self._regression_trajectory = DmpTrajectory(regression[:, 1:])
        # demonstration data
        self._len_demo, self._nb_joints = np.shape(self._y_demo)
        self.T = np.stack((self._y_demo, self._yd_demo, self._ydd_demo), axis=-1)
        # general parameters
        self._tau = self._estimate_time_constant()
        self._c_order = c_order
        # initialize the state variables
        self._z = np.zeros(self._nb_joints)
        self._y = np.zeros(self._nb_joints)
        self._x = np.zeros(self._nb_joints)
        self._v = np.zeros(self._nb_joints)
        self._zd = np.zeros(self._nb_joints)
        self._yd = np.zeros(self._nb_joints)
        self._xd = np.zeros(self._nb_joints)
        self._vd = np.zeros(self._nb_joints)
        self._ydd = np.zeros(self._nb_joints)
        self._y0 = np.zeros(self._nb_joints)
        # the goal state
        self._G = np.zeros(self._nb_joints)
        self._g = np.zeros(self._nb_joints)
        self._gd = np.zeros(self._nb_joints)
        self._s = 1
        # joint dynamics variable
        self.y = np.zeros((self._len_demo, self._nb_joints, 3))
        # initialize some variables for plotting
        self.z_history = np.zeros((self._len_demo, self._nb_joints, 2))
        self.x_history = np.zeros((self._len_demo, self._nb_joints, 2))
        self.v_history = np.zeros((self._len_demo, self._nb_joints, 2))

        if dmp_parameters is None:
            # run optimization
            self._search_space = search_space
            self._set_goal(goal=goal_joints, y0=initial_joints)
            self._optimize_dmp_params()
        else:
            self.set_alpha_z_and_n_rfs(dmp_parameters[0], dmp_parameters[1])

    @classmethod
    def load_dmp(
        cls, dir_path: Path, g_joints: np.ndarray, i_joints: np.ndarray
    ) -> "DynamicMovementPrimitives":
        """
        Creates an DynamicMovementPrimitives with parameters loaded from a dmp param
        file and regression information loaded from a regression file.

        :param dir_path: path of the directory containing the data to load
        :param g_joints: goal robot joints
        :param i_joints: initial robot joints
        :return: DynamicMovementPrimitives with the loaded parameters
        """

        # load regression data
        regression_path = Path.joinpath(dir_path, "regression.npy")
        if not regression_path.exists():
            raise RuntimeError("regression.npy does not exist!")
        reg = np.load(str(regression_path))
        # load dmp parameters
        parameters_path = Path.joinpath(dir_path, "dmp_parameters.json")
        if not parameters_path.exists():
            raise RuntimeError("dmp_parameters,json does not exist!")
        with open(parameters_path, "r") as f:
            data = json.load(f)
        c_order = data["c_order"]
        alpha_z = np.array(data["alpha_z"])
        n_rfs = data["n_rfs"]
        return cls(reg, c_order, g_joints, i_joints, dmp_parameters=(alpha_z, n_rfs),)

    def save_dmp(self, dir_path: Path) -> None:
        """
        Saves dmp information in a zip file. The file contains both the dmp parameters
        and the regression function used to learn the forcing term. If the destination
        folder doesn't exist it creates it.

        :param dir_path: path to the directory where the data will be saved
        """

        # create the parent folder if it does not exist
        if not dir_path.exists():
            try:
                dir_path.mkdir()
            except FileNotFoundError:
                raise FileNotFoundError(
                    "Parent folder does not exists, please check the "
                    "consistency of the provided path!"
                )
        # save regression function to file
        regression_path = dir_path.joinpath("regression.npy")
        if regression_path.exists():
            raise FileExistsError("Not allowed to override an existing file!")
        np.save(str(regression_path), self.regression)
        # save dmp parameters to file
        dmp_param_path = dir_path.joinpath("dmp_parameters.json")
        if dmp_param_path.exists():
            raise FileExistsError("Not allowed to override an existing file!")
        data = {
            "c_order": 1,
            "alpha_z": self._alpha_z.tolist(),
            "n_rfs": int(self._n_rfs),
        }
        with open(dmp_param_path, "w") as f:
            json.dump(data, f)

    def set_alpha_z_and_n_rfs(self, alpha_z: np.ndarray, n_rfs: int) -> None:
        """
        Sets the alpha_z coefficient and the number of radial basis functions used to
        approximate the system forcing term. Then, updates all the quantities that are
        dependent on those variables.

        :param alpha_z: coefficient for critically damped system
        :param n_rfs: number of radial basis functions
        """
        # coefficients for critically damped dynamics
        self._alpha_z = alpha_z
        self._beta_z = alpha_z / 4
        self._alpha_g = alpha_z / 6
        self._alpha_x = alpha_z / 3
        self._alpha_v = alpha_z
        self._beta_v = alpha_z / 4
        # number of radial basis functions
        self._n_rfs = n_rfs
        self._c, self._c_d, self._D = self._rbf_kernels()
        # learn weights of the target forcing term
        self._batch_fit()
        # variables for plotting
        self.psi_history = np.zeros((self._len_demo, self._nb_joints, self._n_rfs))
        self.w_history = np.zeros((self._len_demo, self._nb_joints, self._n_rfs))

    def _reset_state(self) -> None:
        """
        Resets the internal states of the system to the starting point of the trajectory
        evolution.

        """
        # initialize the state variables
        self._z = np.zeros(self._nb_joints)
        self._y = np.zeros(self._nb_joints)
        self._x = np.zeros(self._nb_joints)
        self._v = np.zeros(self._nb_joints)
        self._zd = np.zeros(self._nb_joints)
        self._yd = np.zeros(self._nb_joints)
        self._xd = np.zeros(self._nb_joints)
        self._vd = np.zeros(self._nb_joints)
        self._ydd = np.zeros(self._nb_joints)
        self._y0 = np.zeros(self._nb_joints)
        # the goal state
        self._G = np.zeros(self._nb_joints)
        self._g = np.zeros(self._nb_joints)
        self._gd = np.zeros(self._nb_joints)
        self._s = 1

    def set_alpha_z(self, alpha_z: np.ndarray) -> None:
        """
        Sets alpha_z array value

        :param alpha_z: critically damped coefficient array to set
        """
        self.set_alpha_z_and_n_rfs(alpha_z, self._n_rfs)

    def set_n_rfs(self, n_rfs: int) -> None:
        """
        Sets n_rfs value

        :param n_rfs: number of basis function to use
        """
        self.set_alpha_z_and_n_rfs(self._alpha_z, n_rfs)

    def _error_with(self, alpha_z: np.ndarray, n_rbfs: int) -> float:
        """
        Computes the error with the provided combination of coefficients. The error is
        given by the sum of the root mean squared tracking error with respect to the
        regression function and the distance of the trajectory endpoint with respect to
        the goal reference.

        :param alpha_z: alpha_z candidate value
        :param n_rbfs: number of basis functions candidate value
        :return: the error with the candidate values combination
        """
        self.set_alpha_z_and_n_rfs(alpha_z, n_rbfs)
        dmp_trajectory = self.compute_joint_dynamics(goal=self._G, y_init=self._y0)
        rms_error = dmp_trajectory.rms_error(self._regression_trajectory)
        final_error = np.linalg.norm(dmp_trajectory.joints[-1] - self._G)
        return rms_error + final_error

    def _objective_fct(self, x: list[int]):
        """
        Defines the objective function to evaluate at each method query point.

        :param x: list containing tuples of each dimension search space bounds
        :raises RuntimeError: x should contain 2 integers (1 alpha_z and 1 n_rfs values)
        if the  user wishes to constrain each joint to have the same value for alpha_z,
        or 7 integers (6 alpha_z and 1 n_rfs values) if the user wants to optimize the
        value of alpha_z for each joint separately. If the argument has a different
        dimensionality, then a RuntimeError is raised addressing the issue.
        :return: the objective function value
        """
        # optimization scenario where alpha_z is equal for all joints -> 1j+1n_rfs
        if len(x) == 2:
            alpha_eval = x[0] * np.ones(6)
            n_rbfs_eval = x[1]
        # optimization scenario where alpha_z is optimized for each joint -> 6j+1n_rfs
        elif len(x) == 7:
            alpha_eval = np.array(x[:-1])
            n_rbfs_eval = x[-1]
        else:
            raise RuntimeError(
                f"\nThe search space has the wrong dimensionality!\n"
                f"Argument should have 2 dimensions if we wish to "
                f"constraint equal alpha_z values for each joint or 7 "
                f"dimensions if we wish to optimize alpha_z for each "
                f"joint separately.\nArgument dimension is {len(x)}."
            )
        return self._error_with(alpha_eval, n_rbfs_eval)

    def _optimize_dmp_params(self) -> None:
        """
        Optimizes the parameters by minimising the objective function and sets them.

        """
        res = gp_minimize(
            self._objective_fct,  # the function to minimize
            self._search_space,  # the bounds on each dimension of x
            acq_func="EI",  # the acquisition function
            n_calls=25,  # the number of evaluations of f
        )
        if len(res.x) == 2:
            alpha_z = res.x[0] * np.ones(6)
            n_rfs = res.x[1]
        elif len(res.x) == 7:
            alpha_z = np.array(res.x[:-1])
            n_rfs = res.x[-1]
        else:
            raise RuntimeError(
                "Problem with the optimization process, check the result"
                "dimensionality!"
            )
        self.set_alpha_z_and_n_rfs(alpha_z, n_rfs)

    def compute_joint_dynamics(
        self, goal: np.ndarray, y_init: np.ndarray
    ) -> DmpTrajectory:
        """
        Runs the dynamic movement primitives approach with the learnt basis function
        weights to compute the joint angles dynamics from the initial robot joint angles
        to the target goal

        :param goal: target goal to reach
        :param y_init: initial joint angles position
        :return: the dmp trajectory featuring the joint angles evolution
        """
        self._reset_state()
        # set goal
        self._set_goal(goal=goal, y0=y_init)
        # run dynamic movement primitives
        for i in range(self._len_demo):
            # compute joint dynamics
            self._update_internal_states()
            self.z_history[i] = np.transpose([self._z, self._zd])
            self.y[i] = np.transpose([self._y, self._yd, self._ydd])
            self.x_history[i] = np.transpose([self._x, self._xd])
            self.v_history[i] = np.transpose([self._v, self._vd])
            self.psi_history[i] = self._psi
            self.w_history[i] = self._w
        return DmpTrajectory(self.y[:, :, 0])

    def _compute_derivatives(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Computes the joint angular velocity and angular acceleration from the joint
        angle evolution

        :return: the joint angular velocity and acceleration
        """
        # derivatives
        yd = np.diff(self._y_demo, axis=0) * self.sampling_rate
        ydd = np.diff(yd, axis=0) * self.sampling_rate
        # keep same dimensions adding rows of zeros at the end of motion
        yd = np.vstack((yd, np.zeros(np.shape(yd)[1])))
        ydd = np.vstack(
            (ydd, np.vstack((np.zeros(np.shape(ydd)[1]), np.zeros(np.shape(ydd)[1]))))
        )
        return yd, ydd

    def _estimate_time_constant(self) -> float:
        """
        Estimates the time constant of the system setting a threshold on the velocity of
        the system and counting the time elapsed between the start and the end of the
        identified interval

        :return: the estimated time constant of the dynamical system
        """
        max_velocity = np.max(self._yd_demo)
        threshold = 0.05 * max_velocity
        vel = self._yd_demo[:, 2]
        idx = np.argwhere(np.abs(vel) > threshold)
        return float(self.timestamp[idx[-1]])

    def _compute_f_target(self, G) -> np.ndarray:
        """
        Calculates the target forcing term from the second order differential equation

        :param G: the goal reference target
        :return: the target forcing term
        """
        f_target = self._ydd_demo * self._tau ** 2 - self._alpha_z * (
            self._beta_z * (G - self._y_demo) - self._yd_demo * self._tau
        )
        return f_target

    def _rbf_kernels(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes the rbf kernels used to approximate the forcing term

        :return: the rbf center position, phase velocity at the centers and covariance
        """
        t = np.linspace(0, 1, self._n_rfs).reshape(-1, 1)
        # calculate the rbf center position and phase velocity at the centers
        if self._c_order == 1:
            c = (1 + np.multiply(t, self._alpha_z / 2)) * np.exp(
                np.multiply(t, -self._alpha_z / 2)
            )
            c_d = (-self._alpha_z / 2) * c + np.exp(
                np.multiply(t, -self._alpha_z / 2)
            ) * (self._alpha_z / 2)
        else:
            c = np.exp(np.multiply(t, -self._alpha_x))
            c_d = -self._alpha_x * c

        d = (np.diff(c, axis=0) * 0.55) ** 2
        # calculate basis functions covariance
        D = 1 / np.vstack((d, d[-1]))
        return c, c_d, D

    def _batch_fit(self) -> None:
        """
        Fits the discrete movement primitive to a complete trajectory in batch mode
        """
        # start state as the first state in the trajectory
        y0 = self._y_demo[0, :]
        # goal state as the last state in the trajectory
        goal = self._y_demo[-1, :]
        # set g as continuous variable to filter changes in goal target
        g = y0.copy()
        # initialize the hidden states
        X = np.empty((np.shape(self._y_demo)))
        V = np.empty((np.shape(self._y_demo)))
        G = np.empty((np.shape(self._y_demo)))
        x = np.ones(self._nb_joints)
        v = np.zeros(self._nb_joints)
        # loop over general demonstration length
        for i in range(self._len_demo):
            # store hidden states
            X[i] = x
            V[i] = v
            G[i] = g
            # compute hidden states
            if self._c_order == 1:
                # prevent the forcing term f to create a discontinuity in acceleration
                vd = self._alpha_v * (self._beta_v * (0 - x) - v) / self._tau
                xd = v / self._tau
            else:
                vd = 0
                xd = self._alpha_x * (0 - x) / self._tau

            # filter goal change with first order differential equation
            gd = (goal - g) * self._alpha_g
            x = xd * self.dt + x
            v = vd * self.dt + v
            g = gd * self.dt + g

        # the task target
        self._dG = goal - y0
        # the forcing term target
        f_target = self._compute_f_target(G)
        # data reshaping
        X = X.reshape((self._len_demo, self._nb_joints, 1))
        V = V.reshape((self._len_demo, self._nb_joints, 1))
        f_target = f_target.reshape((self._len_demo, self._nb_joints, 1))
        # basis functions
        PSI = np.exp(
            -0.5
            * (
                (
                    (X * np.ones((1, self._n_rfs)))
                    - np.transpose(
                        self._c.reshape((self._n_rfs, self._nb_joints, 1))
                        * np.ones((1, self._len_demo))
                    )
                )
                ** 2
            )
            * np.transpose(
                self._D.reshape((self._n_rfs, self._nb_joints, 1))
                * np.ones((1, self._len_demo))
            )
        )
        # close form locally weighted regression to determine weights
        if self._c_order == 1:
            self._sx2 = np.sum(((V ** 2) * np.ones((1, self._n_rfs))) * PSI, axis=0)
            self._sxtd = np.sum(
                ((V * f_target) * np.ones((1, self._n_rfs))) * PSI, axis=0
            )

        else:
            self._sx2 = sum(((X ** 2) * np.ones(1, self._n_rfs)) * PSI, 1)
            self._sxtd = sum(((X * f_target) * np.ones(1, self._n_rfs)) * PSI, 1)

        # compute the weights
        self._w = self._sxtd / (self._sx2 + 1.0e-10)

    def _set_goal(self, goal: np.ndarray, y0: np.ndarray) -> None:
        """
        Specifies the joint angles target and the initial robot joint angles state

        :param goal: the goal joint angles
        :param y0: the initial robot joint angles
        """
        assert len(goal) == self._nb_joints, (
            f"expected a goal vector with length equal to the number of robot joints "
            f"({self._nb_joints}), got vector with length {len(goal)}"
        )
        assert len(y0) == self._nb_joints, (
            f"expected a initial joint vector with length equal to the number of "
            f"robot joints ({self._nb_joints}), got vector with length {len(y0)}"
        )
        self._G = goal
        self._x = np.ones(self._nb_joints)
        self._y0 = y0
        self._y = y0
        if self._c_order == 0:
            self._g = self._G
        else:
            self._g = self._y

    def _update_internal_states(self) -> None:
        """
        Updates the dynamical system internal states at time t with the states
        information at time t-1 and the forcing term computed at time t.
        """
        # the weighted sum of the locally weighted regression models
        self._psi = np.transpose(np.exp(-0.5 * ((self._x - self._c) ** 2) * self._D))
        amp = self._s
        if self._c_order == 1:
            init = self._v
        else:
            init = self._x

        forcing_term = (
            np.sum(init.reshape(-1, 1) * self._w * self._psi, axis=1)
            / np.sum(self._psi + 1.0e-10, axis=1)
            * amp
        )

        if self._c_order == 1:
            self._vd = (
                self._alpha_v * (self._beta_v * (0 - self._x) - self._v)
            ) / self._tau
            self._xd = self._v / self._tau
        else:
            self._vd = 0
            self._xd = (self._alpha_x * (0 - self._x)) / self._tau

        self._zd = (
            self._alpha_z * (self._beta_z * (self._g - self._y) - self._z)
            + forcing_term
        ) / self._tau

        self._yd = self._z / self._tau
        self._ydd = self._zd / self._tau
        self._gd = self._alpha_g * (self._G - self._g)
        self._x = self._xd * self.dt + self._x
        self._v = self._vd * self.dt + self._v
        self._z = self._zd * self.dt + self._z
        self._y = self._yd * self.dt + self._y
        self._g = self._gd * self.dt + self._g

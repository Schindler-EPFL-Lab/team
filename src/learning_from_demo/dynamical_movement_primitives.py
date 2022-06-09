import numpy as np

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
        regression_fct=regression.prediction,
        alpha_z=alpha_z,
        n_rfs=10,
        c_order=1,
    )
    dmp.compute_joint_dynamics(goal, y_init)
    ```

    :param regression_fct: regression function extracted from dataset
    :param alpha_z: coefficients for critically damped dynamics
    :param n_rfs: number of basis functions to approximate the forcing term
    :param c_order: order of the dynamical system
    """

    def __init__(
        self, regression_fct: np.ndarray, alpha_z: np.ndarray, n_rfs: int, c_order: int
    ) -> None:

        self.sampling_rate = 1 / (regression_fct[1, 0] - regression_fct[0, 0])
        self.dt = regression_fct[1, 0] - regression_fct[0, 0]
        # extract information
        self.timestamp = regression_fct[:, 0]
        self._y_demo = regression_fct[:, 1:]
        self._yd_demo, self._ydd_demo = self._compute_derivatives()
        # demonstration data
        self._len_demo, self._nb_joints = np.shape(self._y_demo)
        self.T = np.stack((self._y_demo, self._yd_demo, self._ydd_demo), axis=-1)
        # coefficients for critically damped dynamics
        self._alpha_z = alpha_z
        self._beta_z = alpha_z / 4
        self._alpha_g = alpha_z / 6
        self._alpha_x = alpha_z / 3
        self._alpha_v = alpha_z
        self._beta_v = alpha_z / 4
        # general parameters
        self._tau = self._estimate_time_constant()
        self._n_rfs = n_rfs
        self._c_order = c_order
        self._c, self._c_d, self._D = self._rbf_kernels()
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
        # the goal state
        self._G = np.zeros(self._nb_joints)
        self._g = np.zeros(self._nb_joints)
        self._gd = np.zeros(self._nb_joints)
        self._s = 1
        # learn weights of the target forcing term
        self._batch_fit()
        # joint dynamics variable
        self.y = np.zeros((self._len_demo, self._nb_joints, 3))
        # initialize some variables for plotting
        self.z_history = np.zeros((self._len_demo, self._nb_joints, 2))
        self.x_history = np.zeros((self._len_demo, self._nb_joints, 2))
        self.v_history = np.zeros((self._len_demo, self._nb_joints, 2))
        self.psi_history = np.zeros((self._len_demo, self._nb_joints, self._n_rfs))
        self.w_history = np.zeros((self._len_demo, self._nb_joints, self._n_rfs))

    def compute_joint_dynamics(self, goal, y_init) -> DmpTrajectory:
        """
        Runs all the steps to compute the joint angles dynamics

        :param goal: target goal to reach
        :param y_init: initial joint angles position
        :return: the dmp trajectory featuring the joint angles evolution, to run on the
                 robot and achieve the target goal
        """
        # set goal
        self._set_goal(goal=goal, y0=y_init)
        # dynamic movement primitives
        joints_dynamics = self._run_fit()
        joint_angles = joints_dynamics[:, :, 0]
        return DmpTrajectory(joint_angles)

    def _run_fit(self) -> np.ndarray:
        """
        Runs the dynamic movement primitives approach with the learnt basis function
        weights to compute the joint angles dynamics from the initial robot joint angles
        to the target goal

        :return: the joint angles dynamics as an array of shape
                (timestamps x nb_joints x dynamics)
                where dynamics is of length 3 and the indices 0, 1, 2 describe
                respectively the joint angles positions, the joint angular velocities
                and the joint angular accelerations
        """
        self._g = self._y
        self._z = np.zeros(self._nb_joints)

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

        return self.y

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
        self._y = y0
        if self._c_order == 0:
            self._g = self._G

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

import numpy as np
import trcontrol.framework.prob.dists as dists
import trcontrol.framework.prob.channels as channels
import trcontrol.framework.filter.bayes as bayes
import typing

from abc import ABC, abstractmethod
from trcontrol.framework.prob.dists import Distribution
from typing import Callable

StateType = typing.Union[int, np.ndarray]
InputType = typing.Union[int, np.ndarray]
OutputType = typing.Union[int, np.ndarray]


class Policy(ABC):
    def __init__(self, problem: 'ControlProblem'):
        self._problem = problem
        self._solved = False

    @abstractmethod
    def input(self, state: StateType, t: int):
        pass

    @abstractmethod
    def input_channel(self, t: int) -> channels.Channel:
        pass

    @property
    def solved(self) -> bool:
        return self._solved


class ControlProblem(ABC):
    def __init__(self, init_dist: Distribution, horizon: int) -> None:
        self._init_dist = init_dist
        self._horizon = horizon

    @abstractmethod
    def dynamics(self, state: StateType, input: InputType, t: int) -> dists.Distribution:
        pass

    @abstractmethod
    def sensor(self, state: StateType, t: int) -> dists.Distribution:
        pass

    @abstractmethod
    def costs(self, state: StateType, input: InputType, t: int) -> float:
        pass

    @abstractmethod
    def terminal_costs(self, state: StateType) -> float:
        pass

    @property
    def init_dist(self) -> Distribution:
        return self._init_dist

    @property
    def horizon(self) -> int:
        return self._horizon

    @property
    @abstractmethod
    def n_states(self) -> int:
        pass

    @property
    @abstractmethod
    def n_inputs(self) -> int:
        pass

    @property
    @abstractmethod
    def n_outputs(self) -> int:
        pass

    def simulate(self, policy: Policy,
                 filter: Callable[['ControlProblem', OutputType], 'bayes.BayesFilter']) -> (np.ndarray, np.ndarray):
        # Abstract way to determine state size.
        mean = self._init_dist.mean()

        if isinstance(self._init_dist.mean(), np.ndarray):
            n = mean.shape
        else:
            n = 1

        traj = np.zeros(n, self._horizon + 1)
        costs = np.zeros(self._horizon + 1)

        traj[:, 0] = self._init_dist.sample()
        meas = self.sensor(traj[:, 0], 0).sample()
        bf = filter(self, meas)

        for t in range(self.horizon):
            input = policy.input(traj[:, t], t)
            traj[:, t + 1] = self.dynamics(traj[:, t], input, t).sample()

            meas = self.sensor(traj[:, t + 1], t + 1).sample()
            bf.iterate(meas, t)

            costs[t] = self.costs(traj[:, t], input)

        costs[self.horizon + 1] = self.terminal_costs(traj[:, -1])

        return traj, costs


class DSCProblem(ControlProblem):
    def __init__(self, init_dist: dists.FiniteDist, horizon: int):
        super().__init__(init_dist, horizon)

        self._dynamics_tensor = self.create_dynamics_tensor()
        (n, m, _) = self._dynamics_tensor.shape

        self._sensor_tensor = self.create_sensor_tensor()
        self._costs_tensor = self.create_costs_tensor()
        self._terminal_costs_tensor = self.create_terminal_costs_tensor()

    @abstractmethod
    def create_dynamics_tensor(self) -> np.ndarray:
        """
        This function populates an n-by-n-by-m array describing the system dynamics. The entry at (i, j, k) is the
        probability of transitioning from state j to state i via input k.

        :return: The dynamics array.
        """
        pass

    @abstractmethod
    def create_sensor_tensor(self) -> np.ndarray:
        """
        This function populates an l-by-n array describing the sensor model. The entry at (i, j) describes
        the probability of observing output i when the system is in state j.

        :return: The sensor array.
        """
        pass

    @abstractmethod
    def create_costs_tensor(self) -> np.ndarray:
        """
        This function populates the cost model for the problem. The cost model is represented by an n-by-m array where
        the entry at (i, j) describes the cost of using input j in state i.

        :return: The cost model array.
        """
        pass

    @abstractmethod
    def create_terminal_costs_tensor(self) -> np.ndarray:
        """
        This function populates the terminal cost model for the problem. This model is represented by a one dimensional
        array of size n, where the i-th entry describes the cost of terminating in state i.
        :return:
        """
        pass

    def dynamics(self, state: int, input: int, t: int = 0) -> dists.FiniteDist:
        return dists.FiniteDist(self._dynamics_tensor[:, state, input])

    def sensor(self, state: int, t: int = 0) -> dists.FiniteDist:
        return dists.FiniteDist(self._sensor_tensor[:, state])

    def costs(self, state: int, input: int, t: int) -> float:
        return self._costs_tensor[state, input]

    def terminal_costs(self, state: int) -> float:
        return self._terminal_costs_tensor[state]

    @property
    def dynamics_tensor(self):
        return self._dynamics_tensor

    @property
    def sensor_tensor(self):
        return self._sensor_tensor

    @property
    def costs_tensor(self):
        return self._costs_tensor

    @property
    def terminal_costs_tensor(self):
        return self._terminal_costs_tensor

    @property
    @abstractmethod
    def n_states(self) -> int:
        pass

    @property
    @abstractmethod
    def n_inputs(self) -> int:
        pass

    @property
    @abstractmethod
    def n_outputs(self) -> int:
        pass


class LQGProblem(ControlProblem):
    def __init__(self, init_dist: dists.GaussianDist, horizon: int,
                 A: np.ndarray, B: np.ndarray, C: np.ndarray,
                 proc_cov: np.ndarray, meas_cov: np.ndarray,
                 Q: np.ndarray, g: np.ndarray, R: np.ndarray, w: np.ndarray, Qf: np.ndarray):

        self._A = A.copy()
        self._B = B.copy()
        self._C = C.copy()
        self._proc_cov = proc_cov.copy()
        self._meas_cov = meas_cov.copy()
        self._Q = Q.copy()
        self._R = R.copy()
        self._Qf = Qf.copy()
        self._g = g.copy()
        self._w = w.copy()

        super().__init__(init_dist, horizon)

    def dynamics(self, state: np.ndarray, input: np.ndarray, t: int) -> dists.GaussianDist:
        return dists.GaussianDist(self._A[:, :, t] @ state + self._B[:, :, t] @ input,  self._proc_cov)

    def sensor(self, state: np.ndarray, t: int) -> dists.GaussianDist:
        return dists.GaussianDist(self._C[:, :, t] @ state, self._meas_cov)

    def costs(self, state: StateType, input: InputType, t: int) -> float:
        return state.transpose() @ self._Q[:, :, t] @ state + input.transpose() @ self._R[:, :, t] @ input.transpose()

    def terminal_costs(self, state: StateType) -> float:
        return state.transpose() @ self._Qf @ state

    @property
    def n_states(self) -> int:
        return self._A.shape[0]

    @property
    def n_inputs(self) -> int:
        return self._B.shape[1]

    @property
    def n_outputs(self) -> int:
        return self._C.shape[0]


class NLGProblem(ControlProblem):
    @abstractmethod
    def __init__(self, init_dist: dists.GaussianDist, horizon: int,
                 proc_cov: np.ndarray, meas_cov: np.ndarray,
                 Q: np.ndarray, g: np.ndarray, R: np.ndarray, w: np.ndarray, Qf: np.ndarray):
        self._proc_cov = proc_cov.copy()
        self._meas_cov = meas_cov.copy()
        self._Q = Q.copy()
        self._R = R.copy()
        self._Qf = Qf.copy()
        self._g = g.copy()
        self._w = w.copy()

        super().__init__(init_dist, horizon)

    @abstractmethod
    def dynamics(self, state: StateType, input: InputType, t: int) -> dists.Distribution:
        pass

    @abstractmethod
    def linearize_dynamics(self, state: StateType, input: InputType, t: int) -> (np.ndarray, np.ndarray):
        pass

    @abstractmethod
    def sensor(self, state: StateType, t: int) -> dists.Distribution:
        pass

    @abstractmethod
    def linearize_sensor(self, state: StateType, t: int) -> np.ndarray:
        pass

    def costs(self, state: StateType, input: InputType, t: int) -> float:
        return state.transpose() @ self._Q[:, :, t] @ state + input.transpose() @ self._R[:, :, t] @ input.transpose()

    def terminal_costs(self, state: StateType) -> float:
        return state.transpose() @ self._Qf @ state

    @property
    def proc_cov(self) -> np.ndarray:
        return self._proc_cov

    @property
    def meas_cov(self) -> np.ndarray:
        return self._meas_cov

    def linearize(self, state_traj: np.ndarray, input_traj: np.ndarray) -> LQGProblem:
        n = self.n_states
        m = self.n_inputs
        l = self.n_outputs

        A = np.zeros((n, n, self.horizon))
        B = np.zeros((m, n, self.horizon))
        C = np.zeros((l, n, self.horizon))

        for t in range(self._horizon):
            (A[:, :, t], B[:, :, t]) = self.linearize_dynamics(state_traj[:, t], input_traj[:, t], t)
            C[:, :, t] = self.linearize_sensor(state_traj[:, t], t)

        return LQGProblem(self.init_dist, self.horizon, A, B, C, self.proc_cov,
                          self.meas_cov, self._Q, self._g, self._R,self._w, self._Qf)

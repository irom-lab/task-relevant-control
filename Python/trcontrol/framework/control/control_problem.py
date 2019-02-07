import numpy as np
from abc import ABC, abstractmethod
from trcontrol.framework.filter.discrete import DiscreteFilter
import trcontrol.framework.prob.dists as dists
import trcontrol.framework.prob.channels as channels


class ControlProblem(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def simulate(self, init: dists.Distribution, horizon: int) -> (np.ndarray, np.ndarray):
        pass


class DSCProblem(ControlProblem):
    def __init__(self):
        super().__init__()

        self._dynamics = self.create_dynamics()
        (n, m, _) = self._dynamics.shape

        self._policy = np.zeros(m, n)
        self._sensor = self.create_sensor()
        self._costs = self.create_costs()
        self._terminal_costs = self.create_terminal_costs()
        self._state_dist = None

    @abstractmethod
    def create_dynamics(self) -> np.ndarray:
        """
        This function populates an n-by-n-by-m array describing the system dynamics. The entry at (i, j, k) is the
        probability of transitioning from state j to state i via input k.

        :return: The dynamics array.
        """
        pass

    @abstractmethod
    def create_sensor(self) -> np.ndarray:
        """
        This function populates an l-by-n array describing the sensor model. The entry at (i, j) describes
        the probability of observing output i when the system is in state j.

        :return: The sensor array.
        """
        pass

    @abstractmethod
    def create_costs(self) -> np.ndarray:
        """
        This function populates the cost model for the problem. The cost model is represented by an n-by-m array where
        the entry at (i, j) describes the cost of using input j in state i.

        :return: The cost model array.
        """
        pass

    @abstractmethod
    def create_terminal_costs(self) -> np.ndarray:
        """
        This function populates the terminal cost model for the problem. This model is represented by a one dimensional
        array of size n, where the i-th entry describes the cost of terminating in state i.
        :return:
        """
        pass

    @property
    def dynamics(self) -> np.ndarray:
        return self._dynamics

    @property
    def sensor(self) -> np.ndarray:
        return self._sensor

    @property
    def costs(self):
        return self._costs

    @property
    def terminal_costs(self) -> np.ndarray:
        return self._terminal_costs

    @property
    def policy(self) -> np.ndarray:
        return self._policy

    @property
    def state_dist(self) -> np.ndarray:
        return self._state_dist

    def simulate(self, init_dist: dists.FiniteDist, horizon: int) -> (np.ndarray, np.ndarray):
        traj = np.zeros(horizon + 1)
        costs = np.zeros(horizon + 1)
        sensor_channel = channels.DiscreteChannel(self.sensor)

        traj[0] = init_dist.sample()
        meas = sensor_channel.marginal(traj[0]).sample()

        bf = DiscreteFilter(self._dynamics, self._policy, self._sensor, init_dist, meas)

        for t in range(horizon):
            input_dist = dists.FiniteDist(self.policy[:, bf.mle()])
            input = input_dist.sample()
            costs[t] = self.costs[traj[t], input]

            next_state_dist = dists.FiniteDist(self.dynamics[:, traj[t], input])
            traj[t + 1] = next_state_dist.sample()

            meas = sensor_channel.marginal(traj[t + 1])
            bf.iterate(meas)

        costs[horizon + 1] = self.terminal_costs[traj[horizon + 1]]

        return traj, costs

    def solve_value_iter(self, iters: int = 100) -> (np.ndarray, np.ndarray):
        costs = self.costs
        dynamics = self.dynamics
        (n, _, m) = dynamics.shape

        values = np.zeros(n)
        input_idx = np.zeros(n)
        policy = np.zeros(m, n)

        for itr in range(iters):
            for i in range(n):
                possible_future_costs = values @ dynamics[:, i, :] + costs[i, :]
                input_idx[i] = possible_future_costs.argmin()
                values[i] = possible_future_costs[input_idx]

        for i in range(n):
            policy[input_idx[i], i] = 1

        return policy, values

class CSCProblem(ControlProblem):
    pass

import numpy as np
from abc import ABC, abstractmethod
from typing import Union
from scipy.stats import rv_discrete, rv_continuous
from trcontrol.framework.filter.discrete import DiscreteFilter




class ControlProblem(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def simulate(self, init: Union[rv_discrete, rv_continuous], horizon: int) -> (np.ndarray, np.ndarray):
        pass

class DSCProblem(ControlProblem):
    def __init__(self):
        self._dynamics = self.create_dynamics()
        (n, m, _) = self._dynamics.shape

        self._policy = np.zeros(m, n)
        self._sensor = self.create_sensor()
        self._costs = self.create_costs()
        self._terminal_costs = self.create_terminal_costs()
        self._state_dist = np.empty()

    @abstractmethod
    def create_dynamics(self) -> np.ndarray:
        pass

    @abstractmethod
    def create_sensor(self) -> np.ndarray:
        pass

    @abstractmethod
    def create_costs(self) -> np.ndarray:
        pass

    @abstractmethod
    def create_terminal_costs(self) -> np.ndarray:
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

    def simulate(self, init_dist: rv_discrete, horizon: int) -> (np.ndarray, np.ndarray):
        traj = np.zeros(horizon + 1)
        costs = np.zeros(horizon + 1)
        (n_outputs, n_states) = self.sensor.shape

        traj[0] = init_dist.rvs()
        meas = rv_discrete(values=(range(n_outputs), self.sensor[:, traj[0]])).rvs()

        bf = DiscreteFilter(self, init_dist, meas)

        for t in range(horizon):
            input_dist = rv_discrete(values=(range(n_outputs), self.policy[:, bf.mle]))
            input = input_dist.rvs()
            costs[t] = self.costs[traj[t], input]

            next_state_dist = rv_discrete(values=(range(n_outputs), self.dynamics[:, traj[t], input]))
            traj[t + 1] = next_state_dist.rvs()

            meas = rv_discrete(values=(range(n_outputs), self.sensor[:, traj[t + 1]])).rvs()
            bf.iterate(self.state_dist[:, t + 1], meas)

        costs[horizon + 1] = costs[t] = self.terminal_costs[traj[horizon + 1]]

        return (traj, costs)
import numpy as np
import cvxpy as cvx
import trcontrol.framework.prob.channels as channels
from typing import Union
from trcontrol.framework.control.control_problem import ControlProblem, DSCProblem
from trcontrol.framework.prob.dists import FiniteDist, kl
from abc import ABC, abstractmethod

class Policy(ABC):
    def __init__(self, problem: ControlProblem):
        self._problem = problem
        self._solved = False

    @abstractmethod
    def input(self, state: Union[int, np.ndarray], t: int):
        pass

    @property
    def solved(self) -> bool:
        return self._solved


class DiscretePolicy(Policy):
    def __init__(self, problem: DSCProblem):
        self._input_given_state = None
        super().__init__(problem)

    def input(self, state: int, t: int):
        if self._input_given_state is None:
            raise RuntimeError('Need to call DiscretePolicy.solve() before asking for inputs.')
        return np.flatnonzero(self._input_given_state[:, state, t])

    def solve(self, horizon) -> float:
        costs = self._problem.costs
        dynamics = self._problem.dynamics
        (n, _, m) = dynamics.shape

        values = np.zeros((n, horizon + 1))
        values[:, -1] = self._problem.terminal_costs

        self._input_given_state = np.zeros((m, n, horizon))

        for t in range(horizon - 1, -1, -1):
            for i in range(n):
                possible_future_costs = values[:, t + 1] @ dynamics[:, i, :] + costs[i, :]
                input_idx = possible_future_costs.argmin()
                values[i, t] = possible_future_costs[input_idx]
                self._input_given_state[input_idx, i, t] = 1

        self._solved = True

        return values[:, 0] @ self._problem.init_dist.pmf()


class DiscreteTRVPolicy(Policy):
    def __init__(self, problem: DSCProblem, trv_size: int, tradeoff: float):
        (n, _, m) = problem.dynamics.shape

        self._trv_size = trv_size
        self._tradeoff = tradeoff
        self._trv_given_state = np.zeros((trv_size, n, problem.horizon))
        self._input_given_trv = np.zeros((m, trv_size, problem.horizon))

        super().__init__(problem)

    def input(self, state: int, t: int):
        if not self._solved:
            raise RuntimeError('Need to call DiscretePolicy.solve() before asking for inputs.')
        return FiniteDist(self._input_given_state[:, state, t]).sample()

    def solve(self, horizon: int, iters: int=100,
              init_trv_given_state: Union[np.ndarray, None]=None,
              init_input_given_trv: Union[np.ndarray, None]=None):
        costs = self._problem.costs
        dynamics = self._problem.dynamics
        terminal_costs = self._problem.terminal_costs

        (n, _, m) = dynamics.shape
        p = self._trv_size # to be consistent with paper notation

        values = np.zeros((n, horizon + 1))
        values[:, -1] = self._problem.terminal_costs

        state_dist = np.zeros((n, horizon + 1))
        state_dist[:, 0] = self._problem.init_dist.pmf()

        trv_dist = np.zeros((p, horizon))

        if init_trv_given_state is None:
            trv_given_state = np.random.rand(p, n, horizon)
            trv_given_state = trv_given_state / (trv_given_state / trv_given_state.sum(axis=0))
        else:
            trv_given_state = init_trv_given_state.copy()

        if init_input_given_trv is None:
            input_given_trv = np.random.rand(m, p, horizon)
            input_given_trv = input_given_trv / (input_given_trv / input_given_trv.sum(axis=0))
        else:
            input_given_trv = init_input_given_trv.copy()

        obj_hist = np.zeros(iters + 1)
        obj_hist[0] = _objective(dynamics, costs, terminal_costs, self._tradeoff,
                                 trv_given_state, input_given_trv, self._problem.init_dist)
        obj_val = obj_hist[0]


        transitions = np.zeros((n, n))

        for iter in range(iters):

            # Forward Equations
            for t in range(horizon):
                input_given_state = input_given_trv[:, :, t] @ trv_given_state[:, :, t]
                sd = FiniteDist(state_dist[:, t])

                transitions = _forward_eq(dynamics, input_given_state)

                state_dist[:, t + 1] = channels.DiscreteChannel(transitions).marginal(sd).pmf()
                trv_dist[:, t] = channels.DiscreteChannel(trv_given_state[:, :, t]).marginal(sd).pmf()

            # Backward Equations
            for t in range(horizon - 1, -1, -1):
                # TRV de Given State
                for i in range(n):
                    for j in range(p):
                        exponent = -self._tradeoff * ((values[:, t + 1] @ dynamics[:, i, :] @ input_given_trv[:, j, t])
                                                      + (costs[i, :] @ input_given_trv[:, j, t]))
                        trv_given_state[j, i, t] = trv_dist[j, t] * np.exp(exponent)

                trv_given_state[:, :, t] = trv_given_state[:, :, t] / trv_given_state[:, :, t].sum(axis=0)

                # Input Given TRV
                policy = cvx.Variable(m, nonneg=True)
                c = cvx.Parameter(m)
                c.value = np.zeros(m)
                obj = cvx.Minimize(c @ policy)
                cstrs = [cvx.sum(policy) == 1]
                prob = cvx.Problem(obj, cstrs)

                for i in range(p):
                    for j in range(m):
                        c.value[j] = trv_given_state[i, :, t] @ (costs[:, j] * state_dist[:, t]) \
                                     + (values[:, t + 1] @ dynamics[:, :, j]) @ (trv_given_state[i, :, t]
                                                                                  * state_dist[:, t])
                    prob.solve()
                    input_given_trv[:, i, t] = policy.value

                # Value Function
                for i in range(n):
                    input_given_state = input_given_trv[:, :, t] @ trv_given_state[:, i, t]
                    trv_dist[:, t] = trv_given_state[:,:, t] @ state_dist[:, t]

                    values[i, t] = costs[i, :] @ input_given_state \
                        + values[:, t + 1] @ (dynamics[:, i, :] @ input_given_state) \
                        + (1 / self._tradeoff) * kl(FiniteDist(trv_given_state[:, i, t]), FiniteDist(trv_dist[:, t]))

            obj_hist[iter + 1] = _objective(dynamics, costs, terminal_costs, self._tradeoff, trv_given_state,
                                        input_given_trv, self._problem.init_dist)

            if obj_hist[iter + 1] <= obj_val:
                obj_val = obj_hist[iter + 1]
                self._trv_given_state = trv_given_state
                self._input_given_trv = input_given_trv

        self._solved = True

        return obj_val


def _forward_eq(dynamics: np.ndarray, input_given_state: np.ndarray) -> np.ndarray:
    (n, _, m) = dynamics.shape
    transitions = np.zeros((n, n))

    for i in range(n):
        transitions[:, i] = dynamics[:, i, :] @ input_given_state[:, i]

    return transitions


def _objective(dynamics: np.ndarray, costs: np.ndarray, terminal_costs: np.ndarray, tradeoff: float,
               trv_given_state: np.ndarray, input_given_trv: np.ndarray, init_dist: FiniteDist):
    expected = 0
    mi = 0
    state_dist = init_dist
    horizon = trv_given_state.shape[2]

    for t in range(horizon):
        input_given_state = input_given_trv[:, :, t] @ trv_given_state[: ,:, t]
        trv_chan = channels.DiscreteChannel(trv_given_state[:, :, t])
        input_chan = channels.DiscreteChannel(input_given_state)
        dynamics_chan = channels.DiscreteChannel(_forward_eq(dynamics, input_given_state))

        mi += trv_chan.mutual_info(state_dist)
        expected += input_chan.joint(state_dist).pmf() @ costs.ravel(order='F')
        state_dist = dynamics_chan.marginal(state_dist)

    expected += state_dist.pmf() @ terminal_costs

    return expected + (1 / tradeoff) * mi

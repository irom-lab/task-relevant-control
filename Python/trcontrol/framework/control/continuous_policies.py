from typing import Union

import numpy as np

from trcontrol.framework.control.problem import StateType
from trcontrol.framework.prob import channels as channels
from .problem import Policy, LQGProblem, NLGProblem
from ..prob.channels import LGChannel


class LQRPolicy(Policy):
    def __init__(self, problem: LQGProblem):
        self._K = np.zeros((problem.n_inputs, problem.n_states, problem.horizon))
        self._f = np.zeros((problem.n_inputs, problem.horizon))
        super().__init__(problem)

    def input(self, state: Union[int, np.ndarray], t: int):
        return self._K[:, :, t] @ state + self._f[:, t]

    def input_channel(self, t: int) -> LGChannel:
        return LGChannel(self._K[:, :, t], self._f[:, t], np.zeros((self._problem.n_states, self._problem.n_states)))

    def solve(self, ref_states: Union[None, np.ndarray] = None, ref_inputs: Union[None, np.ndarray] = None) -> float:
        if ref_states is None:
            ref_states = np.zeros((self._problem.n_states, self._problem.horizon))

        if ref_inputs is None:
            ref_inputs = np.zeros((self._problem.n_inputs, self._problem.horizon))

        A = self._problem._A
        B = self._problem._B
        Q = self._problem._Q
        R = self._problem._R
        g = self._problem._g

        P = self._problem._Qf
        b = P @ (ref_states[:, -1] - g[:, -1])

        for t in range(self._problem.horizon, -1, -1):
            self._K[:, :, t] = -np.linalg.inv(R[:, :, t] + B[:, :, t].transpose() @ P @ B[:, :, t]) \
                         @ B[:, :, t].transpose() @ P @ A[:, :, t]

            self._f[:, t] = -np.linalg.inv(R[:, :, t] + B[:, :, t].transpose() @ P @ B[:, :, t]) \
                               @ (B[:, :, t].transpose() @ b + R[:, :, t] @ ref_inputs[:, t])

            P = A[:, :, t].transpose() @ P @ (A[:, :, t] - B[:, :, t] @ self._K[:, :, t]) + Q[:, :, t]

            b = (A[:, :, t] - B[:, :, t] @ self._K[:, :, t]).transpose() @ b \
                - self._K[:, :, t].transpose() @ R[:, :, t] @ ref_inputs[:, t] + Q[:, :, t] @ ref_states[:, t]

        return self._problem.init_dist.mean().transpose() @ P @ self._problem.init_dist.mean() \
               + self._problem.init_dist.mean() @ b + np.trace(self._problem.init_dist.cov() @ P)


class ILQRPolicy(Policy):
    def __init__(self, problem: NLGProblem):
        super().__init__(problem)
        self._linearized_policy = None

        self._state_traj = np.zeros((problem.n_states, problem.horizon + 1))
        self._state_traj[:, 0] = problem.init_dist.mean()

        self._input_traj = np.zeros((problem.n_inputs, problem.horizon))

    def input(self, state: StateType, t: int):
        return self._linearized_policy.input(state, t)

    def input_channel(self, t: int) -> channels.Channel:
        pass

    def solve(self, iters: int = 10, initial_inputs: Union[None, np.ndarray] = None, verbose: bool = False):
        obj_val = np.inf
        obj_hist = np.zeros(iters)

        if initial_inputs is not None:
            self._input_traj = initial_inputs.copy()

        nominal_states = np.zeros((self._problem.n_states, self._problem.horizon + 1))

        linearized_policy = None

        for iter in range(iters):
            for t in range(self._problem.horizon):
                if linearized_policy is not None:
                    self._input_traj[:, t] += linearized_policy.input(self._state_traj[:, t]
                                                                      - nominal_states[:, t], t)

                self._state_traj[:, t + 1] = self._problem.dynamics(self._state_traj[:, t],
                                                                    self._input_traj[:, t], t).mean()

                obj_hist[iter] += self._problem.costs(self._state_traj[: t], self._input_traj[:, t])

            obj_hist[iter] += self._problem.terminal_costs(self._state_traj[:, -1])

            nominal_states = self._state_traj.copy()

            if verbose:
                print(f'\t[{iter}] Obj Value: {obj_hist[iter]}')

            if obj_hist[iter] < obj_val:
                obj_val = obj_hist
                self._linearized_policy = linearized_policy

            linearized_policy = LQRPolicy(self._problem.linearize())
            linearized_policy.solve(self._state_traj, self._input_traj)

from typing import Union

import numpy as np

from .problem import Policy, LQGProblem
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

    def solve(self, ref_states: Union[None, np.ndarray] = None, ref_inputs: Union[None, np.ndarray] = None):
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

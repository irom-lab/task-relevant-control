from typing import Union

import numpy as np
import cvxpy as cp

from trcontrol.framework.control.problem import StateType
from trcontrol.framework.prob import channels
from trcontrol.framework.prob import dists
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

        for t in range(self._problem.horizon - 1, -1, -1):
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

    def input(self, state: StateType, t: int) -> np.ndarray:
        return self._linearized_policy.input(state, t)

    def input_channel(self, t: int) -> channels.LGChannel:
        return self._linearized_policy.input_channel(t)

    def solve(self, iters: int = 10, initial_inputs: Union[None, np.ndarray] = None, verbose: bool = False) -> float:
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

                obj_hist[iter] += self._problem.costs(self._state_traj[:, t], self._input_traj[:, t], t)

            obj_hist[iter] += self._problem.terminal_costs(self._state_traj[:, -1])

            nominal_states = self._state_traj.copy()

            if verbose:
                print(f'\t[{iter + 1}] Obj Value: {obj_hist[iter]}')

            if obj_hist[iter] < obj_val:
                obj_val = obj_hist[iter]
                self._linearized_policy = linearized_policy

            linearized_policy = LQRPolicy(self._problem.linearize(self._state_traj, self._input_traj))
            linearized_policy.solve(self._state_traj, self._input_traj)

        self._solved = True

        return obj_val


class ITRVPolicy(Policy):
    def __init__(self, problem: NLGProblem, trv_size: int, tradeoff: float, policy_type: str = 'trv'):
        super().__init__(problem)

        self._trv_size = trv_size
        self._tradeoff = tradeoff
        self._policy_type = policy_type
        self._delta_states = []

    def input(self, state: StateType, t: int) -> np.ndarray:
        if not self._solved:
            raise RuntimeError('Need to call DiscretePolicy.solve() before asking for inputs.')
        if self._policy_type == 'trv':
            return self._K[:, :, t] @ state + self._h[:, t]
        else:
            return self._K[:, :, t] @ dists.GaussianDist(self._C[:, :, t] @ state + self._a[:, t],
                                                         self._Sigma_eta[:, :, t]).sample() + self._h[:, t]

    def input_channel(self, t: int) -> channels.LGChannel:
        pass

    def solve(self, tradeoff: float, iters: int = 10, initial_inputs: Union[None, np.ndarray] = None,
              init_K: Union[None, np.ndarray] = None, init_h: Union[None, np.ndarray] = None,
              init_verbose: bool = False, relinearize_every: int = 5):
        A = np.zeros((self._problem.n_states, self._problem.n_states, self._problem.horizon))
        B = np.zeros((self._problem.n_states, self._problem.n_inputs, self._problem.horizon))


        C = np.dstack([np.eye(self._trv_size, self._problem.n_states)] * self._problem.horizon)
        a = np.zeros(self._trv_size, self._problem.horizon)

        Sigma_eta = np.zeros((self._trv_size, self._trv_size, self._problem.horizon))

        for t in range(self._problem.horizon):
            Sigma_eta[:, :, t] = 0.01 * np.random.rand(self._trv_size, self._trv_size)
            Sigma_eta[:, :, t] = Sigma_eta[:,:, t] * Sigma_eta[:,:, t].transpose()

        if init_K is None:
            K = np.zeros((self._problem.n_inputs, self._trv_size, self._problem.horizon))
        else:
            K = init_K.copy()

        if init_h is None:
            h = np.zeros((self._problem.n_inputs, self._problem.horizon))
        else:
            h = init_h.copy()

        Q = self._problem._Q
        R = self._problem._R

        P = np.zeros(self._problem.n_states, self._problem.n_states, self._problem.horizon + 1)
        b = np.zeros(self._problem.n_states, self._problem.horizon + 1)

        delta_states = [dists.GaussianDist(np.zeros(self._problem.n_states),
                                           np.zeros((self._problem.n_states, self._problem.n_states)))
                            for i in range(self._problem.horizon)]

        delta_inputs = [dists.GaussianDist(np.zeros(self._problem.n_inputs),
                                           np.zeros((self._problem.n_inputs, self._problem.n_inputs)))
                        for i in range(self._problem.horizon)]

        nominal_states = np.zeros((self._problem.n_states, self._problem.horizon + 1))
        nominal_states[:, 0] = self._problem.init_dist.mean()

        if initial_inputs is None:
            nominal_inputs = initial_inputs.copy()
        else:
            nominal_inputs = np.zeros((self._problem.n_inputs, self._problem.horizon))

        for t in range(self._problem.horizon):
            nominal_states[:, t + 1] = self._problem.dynamics(nominal_states[:, t], nominal_inputs[:, t], t).mean()

        relinearize = False
        obj_val = np.inf
        obj_hist = np.zeros(iters)
        mi_total = 0
        expected_cost_total = 0
        best_expected_cost = np.inf
        best_mi = np.inf

        for iter in range(iters):
            # Forward dynamics
            expected_cost_total = 0
            mi_total = 0

            for t in range(self._problem.horizon):
                delta_inputs[t] = dists.GaussianDist(K[:, :, t] @ (C[:, :, t] @ delta_states[t].mean()
                                                                   + a[:, t]) + h[:, t],
                                                     K[:, :, t] @ (C[:, :, t] @ delta_states[t].cov()
                                                                   @ C[:, :, t].transpose() + Sigma_eta[:, :, t])
                                                     @ K[:, :, t].transpose())

                A[:,:, t], B[:,:, t] = self._problem.linearize_dynamics(nominal_states[:, t], nominal_inputs[:, t], t)

                # PEP8? What's that?
                # These bits are far more readable as long lines.
                delta_states[t + 1] = dists.GaussianDist((self._problem.dynamics(nominal_states[:, t] + delta_states[t].mean(),
                                                                                nominal_inputs[:, t] + delta_inputs[t], t) - nominal_states[:, t + 1]).mean(), (A[:, :, t] + B[:, :, t] @ K[:, :, t] @ C[:, :, t]) @ delta_states[t].cov() @ (A[:,:, t] + B[:, :, t] @ K[:, :, t] @ C[:, :, t]).transpose() + (B[:,:, t] @ K[:, :, t]) @ Sigma_eta[:, :, t] @ (B[:, :, t] @ K[:,:, t]).transpose() + self._problem._proc_cov)

                expected_cost_total += self._problem.cost(nominal_states[:, t] + delta_states[t].mean(), nominal_inputs[:, t] + delta_inputs[:, t].mean(), t)

                # TODO: Test mutual info computation.
                mi_total += channels.LGChannel(C[:, :, t], a[:, t], Sigma_eta[:,:, t]).mutual_info(delta_states[t])

            obj_hist[iter] = expected_cost_total + (1 / tradeoff) * mi_total

            if obj_hist[iter] < obj_val:
                self._C = C.copy()
                self._a = a.copy()
                self._K = K.copy()
                self._Sigma_eta = Sigma_eta.copy()
                self._h = h.copy()
                self._nominal_inputs = nominal_inputs
                self._nominal_states = nominal_states

                self._A = A
                self._B = B

                for t in range(self._problem.horizon + 1):
                    self._delta_states[t] = delta_states[t]


                relinearize = True

            if iter % relinearize_every == 0 and relinearize:
                relinearize = False

                for t in range(self._problem.horizon):
                    nominal_inputs[:, t] = nominal_inputs + delta_inputs[t].mean()
                    nominal_states[:, t + 1] = self._problem.dynamics(nominal_states[:, t],
                                                                      nominal_inputs[:, t], t).mean()

                continue

            delta_g = self._problem._g - nominal_states
            delta_w = self._problem._w - nominal_inputs

            P[:, :, -1] = self._problem._Qf
            b[:, -1] = -Q[:, :, -1] @ delta_g[:, -1]

            # Here be dragons...
            for t in range(self._problem.horizon, -1, -1):
                # TRV Given State Map:
                Sigma_eta[:, :, t] = np.linalg.inv(tradeoff * K[:, :, t].transpose() @ (B[:, :, t].transpose() @ P[:, :, t + 1] @ B[:, :, t] + R[:, :, t]) @ K[:, :, t]
                                          + np.linalg.inv(C[:, :, t] @ delta_states[t].cov() @ C[:, :, t].transpose() + Sigma_eta[:, :, t]))

                F = np.linalg.inv(C[:, :, t] @ delta_states[t].cov() @ C[:, :, t].transpose() + Sigma_eta[:, :, t])

                C[:, :, t] = -tradeoff * Sigma_eta[:, :, t] @ K[:, :, t].transpose() @ B[:, :, t].transpose() @ P[:, :, t + 1] @ A[:, :, t]

                a[:, t] = -Sigma_eta[:, :, t] @ (tradeoff * K[:, :, t].transpose() @ B[:, :, t].transpose() @ (b[:, t + 1] + P[:, :, t + 1] @ B[:, :, t] @ h[:, t])
                                        + tradeoff * K[:, :, t].transpose() @ R[:, :, t] @ (h[:, t] - delta_w[:, t]) - F * (C[:, :, t] * delta_states[t].mean() + a[:, :, t]))

                # Input Given TRV Map:
                # First some shorthand
                x_bar = delta_states[t].mean()
                Sigma_x = delta_states[t].cov()
                x_tilde_bar = C[:, :, t] @ x_bar + a[:, t]
                Sigma_x_tilde = C[:, :, t] @ Sigma_x @ C[:, :, t].transpose() + Sigma_eta[:, :, t]

                cpK = cp.Variable((self._problem.n_inputs, self._trv_size))
                cph = cp.Variable(self._problem.n_inputs)

                objective = 0.5 * (cpK @ x_tilde_bar + cph - delta_w[:, t]).T @ R[:, :, t] @ (cpK @ x_tilde_bar + cph - delta_w[:, t]) + 0.5 * cp.trace(cpK.T @ R[:, :, t] @ cpK @ Sigma_x_tilde) + 0.5 @ x_bar @ A[:, :, t].transpose() @ P[:, :, t + 1] @ A[:, :, t] @ x_bar + 0.5 * x_bar.transpose() @ (A[:, :, t].transpose() @ P[:, :, t + 1] @ B[:, :, t] @ cpK @ C[:, :, t] + C[:, :, t].transpose() @ cpK.T @ B[:, :, t].transpose() @ P[:, :, t + 1] @ A[:, :, t]) @ x_bar + x_bar.transpose() @ A[:, :, t].transpose() @ P[:, :, t + 1] @ B[:, :, t] @ cpK @ a[:, t] + x_bar.transpose() @ A[:, :, t].transpose() @ P[:, :, t + 1] @ B[:, :, t] @ cph + 0.5 * x_tilde_bar.transpose() * cpK.T @ B[:, :, t].transpose() @ P[:, :, t + 1] @ B[:, :, t] @ cpK @ x_tilde_bar + x_tilde_bar.transpose() @ cpK.T @ B.transpose() @ P[:, :, t + 1] @ B[:, :, t] @ cph + 0.5 @ cph.T @ B.transpose() @ P[:, :, t + 1] @ B[:, :, t] @ cph + b[:, t + 1].transpose() @ (A[:, :, t] @ x_bar + B[:, :, t] @ cpK @ x_tilde_bar + B[:, :, t] @ cph) + 0.5 * cp.trace(Sigma_x @ A[:, :, t].transpose() @ P[:, :, t + 1] @ A[:, :, t] + Sigma_x @ (A[:, :, t].transpose() @ P[:, :, t + 1] @ B[:, :, t] @ K @ C[:, :, t] + C[:, :, t].transpose() @ cpK.T @ B[:, :, t].transpose() @ P[:, :, t + 1] @ A[:, :, t])) + cp.trace(Sigma_x_tilde @ cpK.T @ B.transpose() @ P[:, :, t + 1] @ B[:, :, t] @ cpK)
                prob = cp.Problem(cp.Minimize(objective), [])
                prob.solve(solver=cp.MOSEK)

                K[:, :, t] = cpK.value.copy()
                h[:, t] = cph.value.copy()

                # Value Function:
                G = C[:, :, t].transpose() @ F @ C[:, :, t]

                P[:, :, t] = Q[:, :, t] + (1 / tradeoff) * G + C.transpose() @ K.transpose() @ R @ K @ C + (A + B @ K @ C).transpose() @ P @ (A + B @ K @ C)

                b[:, t] = (A[:, :, t] + B[:, :, t] @ K[:, :, t] @ C[:, :, t]).transpose() @ P[:, :, t + 1] @ B[:, :, t] @ K[:, :, t] @ a[:, t] - Q[:, :, t] @ delta_g[:, t] - (1 / tradeoff) * G @ delta_states[t].mean() + C[:, :, t].transpose() @ K[:, :, t].transpose() @ R[:, :, t] @ K[:, :, t] @ a[:, t] + (A[:, :, t] + B[:, :, t] @ K[:, :, t] @ C[:, :, t]).transpose() @ b[:, t + 1] + C[:, :, t].transpose() @ K[:, :, t].transpose() @ R[:, :, t] @ h[:, t] - C[:, :, t].transpose() @ K[:, :, t].transpose() @ R[:, :, t] @ delta_w[:, t] + A[:, :, t].transpose() @ P[:, :, t + 1] @ B[:, :, t] @ h[:, t] + C[:, :, t].transpose() @ K[:, :, t].transpose() @ B[:, :, t].transpose() @ P[:, :, t + 1] @ B[:, :, t] @ h[:, t]

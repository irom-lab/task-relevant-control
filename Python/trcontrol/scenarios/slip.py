import numpy as np
from scipy.integrate import solve_ivp


from trcontrol.framework.control.problem import StateType, InputType
from trcontrol.framework.prob import dists as dists
from ..framework.control.problem import NLGProblem


class Slip(NLGProblem):
    def __init__(self, init_dist: dists.GaussianDist, horizon: int,
                 proc_cov: np.ndarray, meas_cov: np.ndarray,
                 Q: np.ndarray, g: np.ndarray, R: np.ndarray,
                 w: np.ndarray, Qf: np.ndarray, mass: float = 1.0, touchdown_radius: float = 1.0,
                 spring_const: float = 300, gravity: float = 9.8):
        super().__init__(init_dist, horizon, proc_cov, meas_cov, Q, g, R, w, Qf)
        self._mass = mass
        self._touchdown_radius = touchdown_radius
        self._spring_const = spring_const
        self._gravity = gravity

    def dynamics(self, state: StateType, input: InputType, t: int) -> dists.Distribution:
        return dists.GaussianDist(slip_return_map(state, input, self), self._proc_cov)

    def linearize_dynamics(self, state: StateType, input: InputType,
                           t: int, delta: float = 1e-2) -> (np.ndarray, np.ndarray):
        A = np.zeros((self.n_states, self.n_states))

        for i in range(self.n_states):
            perturbed = state.copy()
            perturbed[i] += delta
            forward = self.dynamics(perturbed, input, t)

            perturbed = state.copy()
            perturbed[i] -= delta
            reverse = self.dynamics(perturbed, input, t)

            A[:, i] = (forward.mean() - reverse.mean()) / (2 * delta)

        forward = self.dynamics(state, input + delta, t)
        reverse = self.dynamics(state, input - delta, t)

        B = ((forward.mean() - reverse.mean()) / (2 * delta)).reshape((4, 1))

        return A, B

    def sensor(self, state: StateType, t: int) -> dists.Distribution:
        return dists.GaussianDist(state, self._meas_cov)

    def linearize_sensor(self, state: StateType, t: int) -> np.ndarray:
        return np.eye(self.n_states)

    @property
    def mass(self) -> float: return self._mass

    @property
    def touchdown_radius(self) -> float: return self._touchdown_radius

    @property
    def spring_const(self) -> float: return self._spring_const

    @property
    def gravity(self) -> float: return self._gravity

    @property
    def n_states(self) -> int:
        return 4

    @property
    def n_inputs(self) -> int:
        return 1

    @property
    def n_outputs(self) -> int:
        return 4


def slip_return_map(state: np.ndarray, input: np.ndarray, slip: Slip,
                    falling: bool = False, tmax: float = 1.0, max_step: float = 0.0001) -> np.ndarray:
    """
    The Poincare return map for the SLIP model.

    :param state: A 4-vector containing values for the lateral head displacement,
                  touchdown angle, radial velocity, and angular velocity.
    :param input: The change in touchdown angle at the next time step.
    :param slip: The SLIP model instance.
    :param falling: Whether or not the model falling over triggers early termination of model's numerical integration.
    :param tmax: The maximum amount of time allowed for integration
    :param max_step: The maximum step size used for integration
    :return: A 4-vector containing either the next touchdown state of the model or NaNs if the hopper fell over.
    """
    stance_state = np.array([1, state[1], state[2], state[3]])

    stance_events_handle = lambda t, stance_state: stance_events(t, stance_state, slip)
    stance_events_handle.terminal = True

    stance_dynamics_handle = lambda t, stance_state: stance_dynamics(t, stance_state, slip)

    sol = solve_ivp(stance_dynamics_handle, (0, tmax), stance_state, events=stance_events_handle, max_step=max_step)

    final_stance_state = sol.y[:, -1]
    flight_state = stance_to_flight(final_stance_state)

    if sol.t[-1] == tmax or (falling and (flight_state[2] < 0 or flight_state[3] < 0)):
        x = np.empty(5)
        x[:] = np.nan

        return x

    flight_events_handle = lambda t, flight_state: flight_events(t, flight_state, input + state[1], slip)
    flight_events_handle.terminal = True

    flight_dynamics_handle = lambda t, flight_state: flight_dynamics(t, flight_state, slip)
    sol = solve_ivp(flight_dynamics_handle, (sol.t[-1], tmax), flight_state, events=flight_events_handle, max_step=0.0001)

    final_flight_state = sol.y[:, -1]
    next_stance = flight_to_stance(final_flight_state, input + state[1])

    return np.array([state[0] + final_flight_state[0], next_stance[1], next_stance[2], next_stance[3]])


def stance_events(t: float, stance_state: np.ndarray, slip: Slip) -> float:
    return float(not (stance_state[0] > slip.touchdown_radius and stance_state[2] > 0))


def flight_events(t: float, flight_state: np.ndarray, input: float, slip: Slip) -> float:
    return float(not (flight_state[1] < slip.touchdown_radius * np.cos(input) and flight_state[3] < 0))


def flight_to_stance(flight_state: np.ndarray, input: np.ndarray) -> np.ndarray:
    theta0 = input

    r = flight_state[1] / np.cos(theta0)

    return np.array([r, theta0, -flight_state[2] * np.sin(theta0) + flight_state[3] * np.cos(theta0),
                     -(flight_state[2] * np.cos(theta0) + flight_state[3] * np.sin(theta0)) / r])


def stance_to_flight(stance_state: np.ndarray) -> np.ndarray:
    return np.array([-stance_state[0] * np.sin(stance_state[1]),
                     stance_state[0] * np.cos(stance_state[1]),
                     -stance_state[2] * np.sin(stance_state[1])
                     - stance_state[0] * stance_state[3] * np.cos(stance_state[1]),
                     stance_state[2] * np.cos(stance_state[1])
                     - stance_state[0] * stance_state[3] * np.sin(stance_state[1])])


def flight_dynamics(t: float, flight_state: np.ndarray, slip: Slip) -> np.ndarray:
    m = slip.mass
    g = slip.gravity

    return np.array([flight_state[2], flight_state[3], 0, -g / m])


def stance_dynamics(t: float, stance_state: np.ndarray, slip: Slip) -> np.ndarray:
    m = slip.mass
    r0 = slip.touchdown_radius
    k = slip.spring_const
    g = slip.gravity

    return np.array([stance_state[2], stance_state[3],
        k / m * (r0 - stance_state[0]) + stance_state[0] * stance_state[3] ** 2 - g * np.cos(stance_state[1]),
        (g * stance_state[0] * np.sin(stance_state[1])
            - 2 * stance_state[0] * stance_state[2] * stance_state[3]) / (stance_state[0] ** 2)])

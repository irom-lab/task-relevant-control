import numpy as np
from trcontrol.framework.control.control_problem import DSCProblem


class Lava(DSCProblem):
    def __init__(self, length, goal):
        self._length = length
        self._goal = goal
        super().__init__()

    def create_dynamics(self) -> np.ndarray:
        # Order of inputs is left, stay, right.
        dynamics = np.zeros((self._length, self._length, 3))

        for i in range(self._length):
            for k in range(3):
                if i == 0:
                    if k < 2:
                        dynamics[0, i, k] = 1
                    else:
                        dynamics[1, i, k] = 1
                elif 1 <= i < self._length - 1:
                    dynamics[i - 1, i, 0] = 1
                    dynamics[i, i, 1] = 1
                    dynamics[i + 1, i, 2] = 1
                else:
                    dynamics[i, i, k] = 1

        return dynamics

    def create_sensor(self) -> np.ndarray:
        pass

    def create_costs(self) -> np.ndarray:
        costs = np.zeros((self._length, 3))

        costs[:(self._goal + 1), 0] = 1
        costs[(self._goal + 1), 0] = -5
        costs[(self._goal + 2):, 0] = 1

        costs[self._goal, 1] = -5

        costs[:(self._goal - 1), 2] = 1
        costs[self._goal - 1, 2] = -5
        costs[self._goal:, 2] = 1

        return costs

    def create_terminal_costs(self) -> np.ndarray:
        terminal_costs = np.zeros(self._length)

        terminal_costs[self._goal] = -10
        terminal_costs[-1] = 10

        return terminal_costs

import numpy as np
from typing import Union
from trcontrol.framework.control.problem import ControlProblem
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

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

from trcontrol.framework.control.discrete_policies import *

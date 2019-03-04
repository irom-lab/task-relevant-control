from ..control import StateType, ControlProblem
from ..prob.channels import Channel
from abc import ABC, abstractmethod


class Policy(ABC):
    def __init__(self, problem: ControlProblem):
        self._problem = problem
        self._solved = False

    @abstractmethod
    def input(self, state: StateType, t: int):
        pass

    @property
    @abstractmethod
    def input_channel(self) -> Channel:
        pass

    @property
    def solved(self) -> bool:
        return self._solved

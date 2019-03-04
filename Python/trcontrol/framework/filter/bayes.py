import numpy as np
import trcontrol.framework.prob.dists as dists

from abc import ABC, abstractmethod
from typing import Union
from ..control import ControlProblem, Policy, OutputType, StateType


class BayesFilter(ABC):
    def __init__(self, problem: ControlProblem, policy: Policy, init_meas: OutputType) -> None:
        self._policy = policy
        self._belief = self.measurement_update(problem.init_dist, init_meas)

    def belief(self) -> dists.Distribution:
        return self._belief

    @abstractmethod
    def process_update(self, belief: dists.Distribution) -> dists.Distribution:
        pass

    @abstractmethod
    def measurement_update(self, proc_belief: dists.Distribution, meas: OutputType) -> dists.Distribution:
        pass

    @abstractmethod
    def mle(self) -> StateType:
        pass

    def iterate(self, meas: Union[np.ndarray, int]):
        proc_belief = self.process_update(self._belief)
        self._belief = self.measurement_update(proc_belief, meas)

import numpy as np
import trcontrol.framework.prob.dists as dists

from abc import ABC, abstractmethod
from typing import Union
from trcontrol.framework.control import problem


class BayesFilter(ABC):
    def __init__(self, problem: 'problem.ControlProblem', policy: 'problem.Policy',
                 init_meas: 'problem.OutputType') -> None:
        self._policy = policy
        self._belief = self.measurement_update(problem.init_dist, init_meas)

    def belief(self) -> dists.Distribution:
        return self._belief

    @abstractmethod
    def process_update(self, belief: dists.Distribution, t: int) -> dists.Distribution:
        pass

    @abstractmethod
    def measurement_update(self, proc_belief: dists.Distribution, meas: 'problem.OutputType', t: int) -> dists.Distribution:
        pass

    @abstractmethod
    def mle(self) -> 'problem.StateType':
        pass

    def iterate(self, meas: Union[np.ndarray, int], t: int):
        proc_belief = self.process_update(self._belief, t)
        self._belief = self.measurement_update(proc_belief, meas, t)

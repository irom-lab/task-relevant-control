from abc import ABC, abstractmethod
from typing import Union
import trcontrol.framework.prob.dists as dists
import numpy as np


class BayesFilter(ABC):
    def __init__(self, init_dist: dists.Distribution, init_meas: dists.Distribution) -> None:
        self._belief = self.measurement_update(init_dist, init_meas)

    def belief(self) -> dists.Distribution:
        return self._belief

    @abstractmethod
    def process_update(self, belief: dists.Distribution) -> dists.Distribution:
        pass

    @abstractmethod
    def measurement_update(self, proc_belief: dists.Distribution, meas: dists.Distribution) -> dists.Distribution:
        pass

    @abstractmethod
    def mle(self):
        pass

    def iterate(self, meas: dists.Distribution):
        proc_belief = self.process_update(self._belief)
        self._belief = self.measurement_update(proc_belief, meas)

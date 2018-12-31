from abc import ABC, abstractmethod
from typing import Union
from scipy.stats import rv_discrete, rv_continuous


class BayesFilter(ABC):
    def __init__(self, init_dist: Union[rv_discrete, rv_continuous], init_meas) -> None:
        self._belief = self.measurement_update(init_dist, init_dist, init_meas)

    @property
    def belief(self) -> Union[rv_discrete, rv_continuous]:
        return self._belief

    @abstractmethod
    def process_update(self, belief: Union[rv_discrete, rv_continuous]) -> Union[rv_discrete, rv_continuous]:
        pass

    @abstractmethod
    def measurement_update(self, state_dist: Union[rv_discrete, rv_continuous], proc_belief: Union[rv_discrete, rv_continuous], meas) -> Union[rv_discrete, rv_continuous]:
        pass

    @property
    @abstractmethod
    def mle(self):
        pass

    def iterate(self, state_dist: Union[rv_discrete, rv_continuous], meas):
        proc_belief = self.process_update(self.belief)
        self.measurement_update(state_dist, proc_belief, meas)
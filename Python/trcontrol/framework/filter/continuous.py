import numpy as np
import trcontrol.framework.prob.dists as dists
from trcontrol.framework.filter.bayes import BayesFilter
from trcontrol.framework.control.problem import CSCProblem

class KalmanFilter(BayesFilter):
    def __init__(self, cscp: CSCProblem, init_dist: dists.GaussianDist, init_meas: np.ndarray) -> None:
        super().__init__(init_dist, init_meas)
        self._cscp = cscp
        self._time = 0

    def process_update(self, belief: dists.GaussianDist) -> dists.Distribution:
        A, B = self._cscp.linearize(belief.mean, self._time)
        

    def measurement_update(self, proc_belief: dists.GaussianDist, meas: np.ndarray) -> dists.Distribution:
        pass

    def mle(self):
        return self._belief.mean()

    def iterate(self, meas):
        proc_belief = self.process_update(self._belief)
        self._belief = self.measurement_update(proc_belief, meas)

import numpy as np
from trcontrol.framework.filter.bayes import BayesFilter
import trcontrol.framework.prob.dists as dists
import trcontrol.framework.prob.channels as channels


class DiscreteFilter(BayesFilter):
    def __init__(self, dynamics: np.ndarray, policy: np.ndarray, sensor: np.ndarray,
                 init_dist: dists.Distribution, init_meas: int) -> None:
        super().__init__(init_dist, init_meas)
        self._dynamics = dynamics
        self._policy = policy
        self._sensor = sensor

    def mle(self) -> int:
        return np.argmax(self._belief.pmf())

    def process_update(self, belief: dists.FiniteDist) -> dists.FiniteDist:
        (n, _, m) = self._dynamics.shape
        belief_pmf = belief.pmf()
        next_belief_given_input = np.zeros(n, m)

        for i in range(m):
            next_belief_given_input[:, i] = self._dynamics.shape[:, :, i] @ belief_pmf

        input_pmf = self._policy @ belief_pmf

        return dists.FiniteDist(next_belief_given_input @ input_pmf)

    def measurement_update(self, proc_belief: dists.FiniteDist, meas: int) -> dists.FiniteDist:
        channel = channels.DiscreteChannel(self._sensor)

        return channel.posterior(proc_belief, meas)

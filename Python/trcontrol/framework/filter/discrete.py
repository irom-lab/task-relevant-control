import numpy as np
from trcontrol.framework.filter.bayes import BayesFilter
from trcontrol.framework.control.control_problem import DSCProblem
from scipy.stats import rv_discrete
import trcontrol.framework.prob.dists as dists
import trcontrol.framework.prob.channels as channels


class DiscreteFilter(BayesFilter):
    def __init__(self, dscp: DSCProblem, init_dist: dists.Distribution, init_meas: int) -> None:
        super().__init__(init_dist, init_meas)
        self._dscp = dscp

    def mle(self) -> int:
        return np.argmax(self._belief.pmf())

    def process_update(self, belief: dists.FiniteDist) -> dists.FiniteDist:
        dynamics = self._dscp.dynamics
        policy = self._dscp.policy
        (n, _, m) = dynamics.shape
        belief_pmf = belief.pmf()
        next_belief_given_input = np.zeros(n, m)

        for i in range(m):
            next_belief_given_input[:, i] = dynamics[:, :, i] @ belief_pmf

        input_pmf = policy @ belief_pmf

        return dists.FiniteDist(next_belief_given_input @ input_pmf)

    def measurement_update(self, proc_belief: dists.FiniteDist, meas: int) -> dists.FiniteDist:
        channel = channels.DiscreteChannel(self._dscp.sensor)

        return channel.posterior(proc_belief, meas)

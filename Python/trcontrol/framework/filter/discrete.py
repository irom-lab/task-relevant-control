import numpy as np
import trcontrol.framework.prob.dists as dists
import trcontrol.framework.prob.channels as channels

from .bayes import BayesFilter
from ..control import DSCProblem, Policy


class DiscreteFilter(BayesFilter):
    def __init__(self, problem: DSCProblem, policy: Policy, init_meas: int) -> None:
        super().__init__(problem, policy, init_meas)
        self._dynamics = problem.dynamics_tensor
        self._sensor = problem.sensor_tensor

    def mle(self) -> int:
        return np.argmax(self._belief.pmf())

    def process_update(self, belief: dists.FiniteDist) -> dists.FiniteDist:
        (n, _, m) = self._dynamics.shape
        belief_pmf = belief.pmf()
        next_belief_given_input = np.zeros(n, m)

        for i in range(m):
            next_belief_given_input[:, i] = self._dynamics.shape[:, :, i] @ belief_pmf

        input_dist = self._policy.input_channel.marginal(dists.FiniteDist(belief_pmf))

        return dists.FiniteDist(next_belief_given_input @ input_dist.pmf())

    def measurement_update(self, proc_belief: dists.FiniteDist, meas: int) -> dists.FiniteDist:
        channel = channels.DiscreteChannel(self._sensor)

        return channel.posterior(proc_belief, meas)

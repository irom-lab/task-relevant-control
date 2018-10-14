import numpy as np
from trcontrol.framework.filter.bayes import BayesFilter
from trcontrol.framework.control.control_problem import DSCProblem
from scipy.stats import rv_discrete


class DiscreteFilter(BayesFilter):
    def __init__(self, dscp: DSCProblem, init_dist: rv_discrete, init_meas: int) -> None:
        self._dscp = dscp
        super().__init__(init_dist, init_meas)

    @property
    def mle(self) -> int:
        belief = self._belief
        return np.argmax(belief.pmf(np.arange(belief.a, belief.b + 1)))[0]

    def process_update(self, belief: rv_discrete) -> rv_discrete:
        dynamics = self._dscp.dynamics
        policy = self._dscp.policy
        (n, m, _) = dynamics.shape
        belief_pmf = belief.pmf(np.arange(belief.a, belief.b + 1))
        next_belief_given_input = np.zeros(n, m)

        for i in range(m):
            next_belief_given_input[:, i] = dynamics @ belief_pmf

        input_pmf = policy @ belief_pmf
        next_belief_pmf = next_belief_given_input @ input_pmf

        return rv_discrete(values=(np.arange(n), next_belief_pmf))

    def measurement_update(self, state_dist: rv_discrete, proc_belief: rv_discrete, meas: int):
        sensor = self._dscp.sensor
        state_pmf = state_dist.pmf(np.arange(state_dist.a, state_dist.b + 1))
        proc_belief = state_dist.pmf(np.arange(proc_belief.a, proc_belief.b + 1))

        state_given_meas = (sensor * state_pmf) / (sensor @ state_pmf)

        updated_pmf = state_given_meas * proc_belief[:, None]
        updated_pmf = updated_pmf / updated_pmf.sum()

        return rv_discrete(values=(np.arange(updated_pmf.size), updated_pmf))
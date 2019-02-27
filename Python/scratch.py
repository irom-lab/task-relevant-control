import numpy as np
from trcontrol.scenarios.lava import Lava
from trcontrol.framework.control.policies import DiscreteTRVPolicy, DiscretePolicy
from trcontrol.framework.prob.dists import FiniteDist

init_dist = FiniteDist(np.array([0.3, 0.4, 0, 0.3, 0]))

lava = Lava(5, 2, init_dist, 5)

dp = DiscretePolicy(lava)
dp.solve(5)

dtp = DiscreteTRVPolicy(lava, 5, 0.01)
dtp.solve(5, init_input_given_trv=dp._input_given_state,
          init_trv_given_state=np.repeat(np.eye(5)[:, :, np.newaxis], 5, axis=2))

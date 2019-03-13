import numpy as np
from trcontrol.scenarios.lava import Lava
from trcontrol.framework.control.discrete_policies import DiscreteTRVPolicy, DiscretePolicy
from trcontrol.framework.prob.dists import FiniteDist

np.random.seed(0)

init_dist = FiniteDist(np.array([0.3, 0.4, 0, 0.3, 0]))

lava = Lava(5, 2, init_dist, 5)

dp = DiscretePolicy(lava)
dp.solve()

dtp = DiscreteTRVPolicy(lava, 5, 1)
dtp.solve(5, 20, True)

for t in range(5):
    print(dtp._trv_given_state[:, :, t].round(decimals=3))

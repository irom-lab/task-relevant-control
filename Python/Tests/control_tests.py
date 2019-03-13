from trcontrol.scenarios.lava import Lava
from trcontrol.framework.control.discrete_policies import DiscretePolicy
from trcontrol.framework.prob import dists

import unittest
import numpy as np


class DSCProblemTests(unittest.TestCase):
    def test_value_iter(self):
        lava = Lava(5, 2, dists.FiniteDist(np.array([1, 0, 0, 0, 0])), 10)
        policy = DiscretePolicy(lava)

        val = policy.solve()

        for t in range(9):
            self.assertTrue(np.allclose(policy._input_given_state[:, :, t], np.array([[0, 0, 0, 1, 0],
                                                                                      [0, 0, 1, 0, 1],
                                                                                      [1, 1, 0, 0, 0]])))

        self.assertTrue(np.allclose(policy._input_given_state[:, :, -1], np.array([[0, 0, 0, 1, 0],
                                                                                   [1, 0, 1, 0, 1],
                                                                                   [0, 1, 0, 0, 0]])))

        self.assertTrue(np.allclose(val, -54))


if __name__ == '__main__':
    unittest.main()

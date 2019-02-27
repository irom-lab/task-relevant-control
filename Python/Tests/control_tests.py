from trcontrol.scenarios.lava import Lava
import unittest
import numpy as np


class DSCProblemTests(unittest.TestCase):
    def test_value_iter(self):
        g = Lava(5, 2)

        (policy, values) = g.solve_value_iter(10)

        for t in range(9):
            self.assertTrue(np.allclose(policy[:, :, t], np.array([[0, 0, 0, 1, 0],
                                                                   [0, 0, 1, 0, 1],
                                                                   [1, 1, 0, 0, 0]])))

        self.assertTrue(np.allclose(policy[:, :, -1], np.array([[0, 0, 0, 1, 0],
                                                                [1, 0, 1, 0, 1],
                                                                [0, 1, 0, 0, 0]])))

        self.assertTrue(np.allclose(values[:, 0], np.array([-54, -60, -60, -60, 10])))


if __name__ == '__main__':
    unittest.main()

from trcontrol.scenarios.gaze import Gaze
import unittest
import numpy as np


class DSCProblemTests(unittest.TestCase):
    def test_value_iter(self):
        g = Gaze(5, 2)

        (policy, values) = g.solve_value_iter()
        self.assertTrue(np.allclose(policy, np.array([[0, 0, 0, 1, 0], [0, 0, 1, 0, 1], [1, 1, 0, 0, 0]])))


if __name__ == '__main__':
    unittest.main()

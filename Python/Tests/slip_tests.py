import unittest
import numpy as np
from trcontrol.scenarios.slip import slip_return_map

class DummySlip:
    pass

class SlipTests(unittest.TestCase):
    def test_slip_return_map(self):
        slip_steady_state = np.array([0.9998, 0.3927, -3.2733, -6.7881])
        slip = DummySlip()
        slip.mass = 1
        slip.touchdown_radius = 1
        slip.spring_const = 300
        slip.gravity = 9.8

        self.assertTrue(np.allclose(slip_return_map(slip_steady_state, 0, slip),
                                    np.array([2.0222, 0.3927, -3.2608, -6.7955]), rtol=0.001))


import unittest
import numpy as np

from trcontrol.scenarios.slip import slip_return_map, Slip
from trcontrol.framework.prob import dists
from trcontrol.framework.control.continuous_policies import ILQRPolicy

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

    def test_ilqr(self):
        g = np.zeros((4, 4))
        g[0, -1] = 3.2
        Qf = np.zeros((4, 4))
        Qf[0, 0] = 1

        s = Slip(init_dist=dists.GaussianDist(np.array([0, 0.3927, -3.2733, -6.7881]), 1e-3 * np.eye(4)),
                 horizon=3,
                 proc_cov=1e-4 * np.diag(np.array([1, 0.1, 0.5, 0.5])),
                 meas_cov=1e-4 * np.eye(4),
                 Q=np.zeros((4, 4, 3)),
                 g=g,
                 R=10 * np.ones((1, 1, 3)),
                 w=np.zeros((1, 3)),
                 Qf=Qf)

        policy = ILQRPolicy(s)
        policy.solve(iters=5, verbose=True)

        self.assertAlmostEqual(policy._state_traj[0, -1], g[0, -1], places=3)


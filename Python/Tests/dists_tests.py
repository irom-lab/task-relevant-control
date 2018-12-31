import unittest
import numpy as np
import trcontrol.framework.prob.dists as dists


class FiniteDistTests(unittest.TestCase):
    def test_mean(self):
        pmf = np.array([0.2, 0.4, 0.4])
        dist = dists.FiniteDist(pmf)

        self.assertAlmostEqual(dist.mean(), 1.2)

    def test_cov(self):
        pmf = np.array([0.2, 0.4, 0.4])
        dist = dists.FiniteDist(pmf)

        self.assertAlmostEqual(dist.cov(), 0.56)

    def test_pmf(self):
        pmf = np.array([0.2, 0.4, 0.4])
        dist = dists.FiniteDist(pmf)

        for i in range(3):
            self.assertEqual(dist.pmf(i), pmf[i])

        self.assertEqual(dist.pmf(-1), 0)
        self.assertEqual(dist.pmf(500), 0)
        self.assertTrue(np.allclose(dist.pmf(), pmf))

    def test_mle(self):
        pmf = np.array([0.2, 0.39, 0.41])
        dist = dists.FiniteDist(pmf)

        self.assertEqual(dist.mle(), 2)

if __name__ == '__main__':
    unittest.main()
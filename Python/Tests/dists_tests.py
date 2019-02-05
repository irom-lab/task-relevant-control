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

        self.assertTrue(np.allclose(dist.pmf(), pmf))

    def test_numel(self):
        pmf = np.array([0.2, 0.4, 0.4])
        dist = dists.FiniteDist(pmf)

        for i in range(3):
            self.assertEqual(dist.pmf(i), pmf[i])

        self.assertEqual(dist.numel(), 3)


# The class GaussianDist is really just a wrapper, so I'm
# skipping writing tests for now.
class GaussianDistTests(unittest.TestCase):
    pass


class KLTests(unittest.TestCase):
    def test_discrete(self):
        pmf1 = np.array([0.36, 0.48, 0.16])
        pmf2 = np.array([1, 1, 1]) / 3

        dist1 = dists.FiniteDist(pmf1)
        dist2 = dists.FiniteDist(pmf2)

        self.assertAlmostEqual(dists.kl(dist1, dist2), 0.0852996)
        self.assertAlmostEqual(dists.kl(dist2, dist1), 0.097455)

    def test_gaussian(self):
        pass

if __name__ == '__main__':
    unittest.main()
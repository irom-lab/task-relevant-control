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

    def test_sample(self):
        np.random.seed(0)
        pmf = np.array([0.2, 0.4, 0.4])
        dist = dists.FiniteDist(pmf)
        n = 10000

        samples = dist.sample(n)
        sample_dist = np.array([(samples == 0).sum() / n,
                                (samples == 1).sum() / n,
                                (samples == 2).sum() / n])

        self.assertTrue(np.allclose(sample_dist, pmf, atol=1e-2))


# The class GaussianDist is really just a wrapper, so I'm
# skipping writing most tests for now.
class GaussianDistTests(unittest.TestCase):
    def test_sample(self):
        np.random.seed(0)
        n = 100000
        dist1 = dists.GaussianDist(np.ones(1), 2 * np.eye(1))
        dist2 = dists.GaussianDist(np.zeros((3, 1)), np.eye(3))

        samples = dist1.sample(n)
        mean = samples.sum() / n
        var = (1 / n) * ((samples - mean) ** 2).sum()

        self.assertAlmostEqual(mean, 1, 2)
        self.assertAlmostEqual(var, 2, 1)

        samples = dist2.sample(n)
        mean = samples.sum(axis=1) / n

        deviation = samples - (mean.reshape((3, 1)) @ np.ones((1, n)))
        cov = (1 / (n - 1)) * (deviation @ deviation.transpose())

        self.assertTrue(np.allclose(mean, np.zeros((3, 1)), atol=0.01))
        self.assertTrue(np.allclose(cov, np.eye(3), atol=0.01))



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
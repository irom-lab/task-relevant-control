import unittest
import trcontrol.framework.prob.channels as channels
import trcontrol.framework.prob.dists as dists
import numpy as np

class DiscreteChannelTests(unittest.TestCase):
    def test_joint(self):
        channel = channels.DiscreteChannel(np.array([[0.2, 0.5], [0.8, 0.5]]))
        input = dists.FiniteDist(np.array([0.5, 0.5]))

        joint = channel.joint(input)

        self.assertTrue(np.allclose(joint.pmf(), np.array([0.1, 0.25, 0.4, 0.25])))

        self.assertAlmostEqual(joint.pmf((0, 1), (2, 2)), 0.25)
        self.assertAlmostEqual(joint.pmf((1, 0), (2, 2)), 0.4)

    def test_conditional(self):
        channel = channels.DiscreteChannel(np.array([[0.2, 0.5], [0.8, 0.5]]))

        self.assertTrue(np.allclose(channel.conditional(0).pmf(), np.array([0.2, 0.8])))
        self.assertTrue(np.allclose(channel.conditional(1).pmf(), np.array([0.5, 0.5])))

    def test_marginal(self):
        channel = channels.DiscreteChannel(np.array([[0.2, 0.5], [0.8, 0.5]]))
        input = dists.FiniteDist(np.array([0.5, 0.5]))

        marginal = channel.marginal(input)

        self.assertTrue(np.allclose(marginal.pmf(), np.array([0.35, 0.65])))

    def test_posterior(self):
        prior = dists.FiniteDist(np.array([0.4, 0.6]))
        channel = channels.DiscreteChannel(np.array([[0.5, 1],[0.5, 0]]))
        posterior = channel.posterior(prior, 0)

        self.assertTrue(np.allclose(posterior.pmf(), np.array([0.25, 0.75])))

    def test_mutual_info(self):
        input = dists.FiniteDist(np.array([1/2, 1/4, 1/8, 1/8]))

        channel = channels.DiscreteChannel(np.array([[1/4, 1/4, 1/4, 1/4],
                                                     [1/8, 1/2, 1/4, 1/4],
                                                     [1/8, 1/4, 1/2, 1/2],
                                                     [1/2,   0,   0,   0]]))
        mi = channel.mutual_info(input, 2)

        self.assertAlmostEqual(mi, 3/8)


    # At this time, we do not use channel capacity, so it is untested.
    def test_channel_capacity(self):
        pass

class LGChannelTests(unittest.TestCase):
    def test_joint(self):
        pass

    def test_conditional(self):
        pass

    def test_marginal(self):
        pass

    def test_posterior(self):
        np.random.seed(0)

        # This example is taken from the end of: http://web.stanford.edu/class/ee363/lectures/estim.pdf
        channel = channels.LGChannel(np.array([[np.cos(np.pi / 6), np.sin(np.pi / 6)]]),
                                     np.ones((1)), np.eye(1))
        input = dists.GaussianDist(np.array([1, 1]), np.array([[2 ** 2, 0], [0, 0.5 ** 2]]))

        sample_input = input.sample()
        conditional = channel.conditional(sample_input)
        sample_output = conditional.sample()

        posterior = channel.posterior(input, sample_output)

        self.assertTrue(np.allclose(posterior.cov(), np.array([[1.046, -0.107],
                                                               [-0.107, 0.246]]), atol=0.01))

        # One more time with a different channel to test the mean

        channel = channels.LGChannel(np.array([[np.cos(i * np.pi / 12), np.sin(i * np.pi / 12)] for i in range(24)]),
                                     np.ones((24)), np.eye(24))
        input = dists.GaussianDist(np.array([1, 1]), np.array([[2 ** 2, 0], [0, 0.5 ** 2]]))

        sample_input = input.sample()
        conditional = channel.conditional(sample_input)
        sample_output = conditional.sample()

        posterior = channel.posterior(input, sample_output)

        self.assertTrue(np.allclose(posterior.mean(), sample_input.flatten(), atol=0.6))


    def test_mutual_info(self):
        pass


if __name__ == '__main__':
    unittest.main()
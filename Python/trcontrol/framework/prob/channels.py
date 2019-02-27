import numpy as np
import cvxpy as cvx

from trcontrol.framework.prob import dists
from abc import ABC, abstractmethod
from typing import Union


class Channel(ABC):
    """
    An abstract base class representing a communication channel, i.e. a Markov chain X -> Y.
    """
    def __init__(self) -> None:
        pass

    @abstractmethod
    def joint(self, chan_input: dists.Distribution) -> dists.Distribution:
        """
        Computes the joint distribution for (X, Y).

        :param chan_input: The distribution of X.
        :return: The joint distribution resulting from the channel properties and input.
        """
        pass

    @abstractmethod
    def conditional(self, chan_input: Union[int, np.ndarray]) -> dists.Distribution:
        """
        Computes the distribution of Y assuming X = x.

        :param chan_input: The value of x.
        :return: The conditional distribution of Y for the given input value.
        """
        pass

    @abstractmethod
    def marginal(self, prior: dists.Distribution) -> dists.Distribution:
        """
        Computes the distribution of Y resulting from a prior over X.

        :param prior: The assumed prior on X.
        :return: The marginal distribution of Y.
        """
        pass

    @abstractmethod
    def posterior(self, prior: dists.Distribution, output: Union[int, np.ndarray]) -> dists.Distribution:
        """
        Computes the posterior distribution of the channel, i.e. the value of X given Y = y.

        :param prior: The assumed prior distribution on X.
        :param output: The value of y.
        :return: The posterior distribution.
        """
        pass

    @abstractmethod
    def mutual_info(self, chan_input: dists.Distribution) -> float:
        """
        Calculates the mutual information between X and Y.

        :param chan_input: The distribution of X.
        :return: The mutual information value.
        """
        pass

    @abstractmethod
    def chan_capacity(self) -> float:
        """
        Finds the capacity of the channel, i.e. the supremum over the mutual information between X and Y.

        :return: The channel capacity value.
        """
        pass


class DiscreteChannel:
    """
    This class models a finite memoryless channel. That is, X and Y are both finite random variables defined over
    sets with cardinality n and m respectively.
    """
    def __init__(self, y_given_x: np.ndarray) -> None:
        """
        Constructs a new discrete channel.

        :param y_given_x: An m-by-n  right stochastic matrix where the value of (j, i) is the probability of Y = j
        given X = i.
        """

        super().__init__()
        self._y_given_x = y_given_x.copy()

    def joint(self, chan_input: dists.FiniteDist) -> dists.FiniteDist:
        """
        Computes the joint distribution for (X, Y).

        :param chan_input: A finite distribution over a set of size n.
        :return: A finite distribution over a set of size n * m. The probability of the event {X = i, Y = j} can be
        accessed using a pmf method call with val=(i, j), shape=(m, n).
        """
        return dists.FiniteDist((self._y_given_x * chan_input.pmf()).flatten())

    def conditional(self, chan_input: int) -> dists.FiniteDist:
        """
        Computes the distribution of Y assuming X = x.

        :param chan_input: The integer x.
        :return: A finite distribution over a set of size m representing the resulting distribution of Y.
        """
        return dists.FiniteDist(self._y_given_x[:, chan_input])

    def marginal(self, prior: dists.FiniteDist) -> dists.FiniteDist:
        """
        Computes the distribution of Y resulting from a prior over X.

        :param prior: The assumed prior on X.
        :return: The marginal distribution of Y.
        """

        return dists.FiniteDist(self._y_given_x @ prior.pmf())

    def posterior(self, prior: dists.FiniteDist, output: int) -> dists.FiniteDist:
        """
        Computes the posterior distribution over X given Y = y.

        :param prior: A finite distribution over n elements representing assumed prior distribution over X.
        :param output: The index of the observed value y.
        :return: A finite distribution over n elements representing the posterior distribution over X.
        """

        return dists.FiniteDist((self._y_given_x[output, :] * prior.pmf()) / self.marginal(prior).pmf(output))

    def mutual_info(self, chan_input: dists.FiniteDist, base: float='e') -> float:
        """
        Calculates the mutual information between X and Y.

        :param chan_input: A finite distribution over n elements representing the assumed distribution over X.
        :return: The mutual information.
        """
        n, m = self._y_given_x.shape
        joint = self.joint(chan_input).pmf(shape=(n, m))
        marginal = self.marginal(chan_input).pmf()

        denom = marginal.reshape((-1, 1)) * chan_input.pmf()
        inside_log = np.zeros(denom.shape)

        denom_nonzeros = denom != 0
        joint_nonzeros = joint != 0

        inside_log[denom_nonzeros] = joint[denom_nonzeros] / denom[denom_nonzeros]

        pointwise = np.zeros((n, m))

        if base == 'e':
            pointwise[joint_nonzeros] = joint[joint_nonzeros] * np.log(inside_log[joint_nonzeros])
        elif base == 2:
            pointwise[joint_nonzeros] = joint[joint_nonzeros] * np.log2(inside_log[joint_nonzeros])
        else:
            raise TypeError("Currently only handles base=2 or base='e'.")

        return pointwise.sum()

    def chan_capacity(self) -> float:
        """
        Finds the capacity of the channel, i.e. the supremum over the mutual information between X and Y, by solving
        a convex optimization problem using CVXPY.

        :return: The channel capacity value.
        """
        n, m = self._y_given_x.shape

        x = cvx.Variable(n)
        y = self._y_given_x @ x
        c = (self._y_given_x * np.log(self._y_given_x)).sum(axis=None)
        mi = c * x + cvx.sum(cvx.entr(y))

        objective = cvx.Minimize(-mi)
        constraints = [cvx.sum(x) == 1, x >= 0]
        problem = cvx.Problem(objective, constraints)
        problem.solve()

        if problem.status == 'optimal':
            return problem.value
        else:
            return np.nan


class LGChannel(Channel):
    """
    This class models a memoryless linear (really affine) Gaussian channel, i.e. for an n-dimensional Gaussian random
    variable X, Y is given by the equation Y = A X + b + eta, where A is an m-by-n array, b is an m-by-1 vector,
    and eta is a zero-mean m-dimensional Gaussian random variable.
    """
    def __init__(self, linear: np.ndarray, affine: np.ndarray, cov: np.ndarray) -> None:
        """
        Constructs a new linear channel.

        :param linear: The linear term, A, which is an m-by-n array.
        :param affine: The affine term, b, which is a size m vector or m-by-1 array.
        :param cov: The covariance of eta, which is an m-by-m array.
        """
        if linear.ndim != 2:
            raise ValueError('The linear component must be 2D.')

        if affine.ndim > 2 or (affine.ndim == 2 and affine.shape[1] != 1):
            raise ValueError('The affine component must be 1D or a m-by-1 vector.')


        super().__init__()
        self._A = linear.copy()
        self._b = affine.flatten()
        self._cov = cov.copy()

    def joint(self, chan_input: dists.GaussianDist) -> dists.GaussianDist:
        """
        Computes the joint Gaussian distribution for (X, Y).

        :param chan_input: The Gaussian distribution of X.
        :return: A the Gaussian distribution of (X, Y) with n + m variables.
        """
        mean = self._A @ chan_input.mean() + self._b
        cov = self._A @ chan_input.cov() @ self._A.transpose() + self._cov

        return dists.GaussianDist(np.block([chan_input.mean(), mean]), np.block([[chan_input.cov(), chan_input.cov() @ self._A.transpose()],
                                                                                 [self._A.transpose() @ chan_input.cov(), cov]]))

    def conditional(self, chan_input: np.ndarray) -> dists.GaussianDist:
        """
        Computes the distribution of Y assuming X = x.

        :param chan_input: The integer x.
        :return: A Gaussian distribution over a set of size m representing the resulting distribution of Y.
        """
        mean = self._A @ chan_input.flatten() + self._b
        cov = self._cov

        return dists.GaussianDist(mean, cov)

    def marginal(self, prior: dists.GaussianDist) -> dists.GaussianDist:
        """
        Computes the distribution of Y resulting from a prior over X.

        :param prior: The assumed Gaussian prior on X.
        :return: The marginal Gaussian distribution of Y.
        """
        return dists.GaussianDist(self._A @ prior.mean() + self._b,
                                  self._A @ prior.cov() @ self._A.transpose() + self._cov)

    def posterior(self, prior: dists.GaussianDist, output: np.ndarray) -> dists.GaussianDist:
        """
        Computes the posterior distribution over X given Y = y.

        Reference: https://web.stanford.edu/class/ee363/lectures/estim.pdf

        :param prior: A finite distribution over n elements representing assumed prior distribution over X.
        :param output: An m-vector representing the observed value of y.
        :return: A finite distribution over n elements representing the posterior distribution over X.
        """

        B = prior.cov() @ self._A.transpose() @ np.linalg.inv(self._A @ prior.cov() @ self._A.transpose() + self._cov)

        output_mean = self._A @ prior.mean() + self._b

        return dists.GaussianDist(prior.mean() + B @ (output.flatten() - output_mean),
                                  np.linalg.inv(self._A.transpose() @ np.linalg.inv(self._cov) @ self._A +
                                                np.linalg.inv(prior.cov())))

    def mutual_info(self, chan_input: dists.GaussianDist) -> float:
        """
        Calculates the mutual information between X and Y.

        :param chan_input: An m-dimensional Gaussian distribution representing the assumed distribution over X.
        :return: The mutual information.
        """
        y_cov = self._A @ chan_input.cov() @ self._A.transpose() + self._cov
        return np.log(np.linalg.det(y_cov)) - np.log(np.linalg.det(self._cov))

    def chan_capacity(self) -> float:
        """
        Finds the capacity of the channel, i.e. the supremum over the mutual information between X and Y, by solving
        a convex optimization problem using CVXPY.

        :return: The channel capacity value.
        """
        n = self._A.shape[1]
        cov = cvx.Variable((n, n))

        y_cov = self._A @ cov @ self._A.transpose() + self._cov
        objective = cvx.log_det(y_cov) - np.log(np.linalg.det(self._cov))

        problem = cvx.Problem(objective, [])
        problem.solve()

        if problem.status == 'optimal':
            return problem.value
        else:
            return np.nan

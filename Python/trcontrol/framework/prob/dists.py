import numpy as np
from scipy.stats import multivariate_normal
from abc import ABC, abstractmethod
from typing import Union, Tuple, Optional


class Distribution(ABC):
    """
    An abstract base class representing a distribution.
    """

    def __init__(self) -> None:
        pass

    @abstractmethod
    def pmf(self, val: Optional[int] = None) -> Union[float, np.ndarray]:
        """
        For discrete distributions, returns the probability mass associated
        with each value in the support.

        :param val: The value at which to evaluate the PMF or None (default).

        :return: The value of the PMF at val or the whole PMF vector if val is None.
        """
        pass

    @abstractmethod
    def pdf(self, val: np.ndarray) -> float:
        """
        For continuous distributions, returns the density associated with a value in the
        distribution's support.

        :param val: The value at which to evaluate the probability density function (PDF).

        :return: The value of the PDF.
        """
        pass

    @abstractmethod
    def mean(self) -> Union[float, np.ndarray]:
        """
        Computes the mean of the distribution.

        :return: The mean of the distribution.
        """
        pass

    @abstractmethod
    def cov(self) -> Union[float, np.ndarray]:
        """
        Computes the covariance of the distribution.

        :return: The covariance of the distribution.
        """
        pass

    @abstractmethod
    def sample(self, n: int = 1) -> Union[int, float, np.ndarray]:
        """
        Samples n points from the distribution
        :param n: Number of points to sample (default is 1)
        :return: If n = 1, returns the value of the sample.
            When n > 1, returns an ndarray containing the samples
        """
        pass


class DiscreteDist(Distribution):
    """
    An abstract base class for discrete distributions.

    The primary purpose of this class is to disallow use of functions only defined for continuous distributions (i.e.
    the PDF).
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def pmf(self, val: Optional[int] = None) -> Union[float, np.ndarray]:
        pass

    def pdf(self, val: np.ndarray): raise NotImplementedError('Discrete distributions do not have PDFs.')

    @abstractmethod
    def mean(self) -> float:
        pass

    @abstractmethod
    def cov(self) -> float:
        pass

    @abstractmethod
    def sample(self, n: int = 1) -> Union[int, np.ndarray]:
        pass


class ContinuousDist(Distribution):
    """
    An abstract base class for continuous distributions.

    The primary purpose of this class is to disallow use of functions only defined for discrete distributions (i.e.
    the PMF).
    """

    def __init__(self):
        super().__init__()

    def pmf(self, val: Optional[int] = None): raise NotImplementedError('Continuous distributions do not have PMFs.')

    @abstractmethod
    def pdf(self, val: np.ndarray) -> float:
        pass

    @abstractmethod
    def mean(self) -> Union[float, np.ndarray]:
        pass

    @abstractmethod
    def cov(self) -> Union[float, np.ndarray]:
        pass

    @abstractmethod
    def sample(self, n: int = 1) -> Union[float, np.ndarray]:
        pass


class FiniteDist(DiscreteDist):
    """
    A probability distribution defined over a finite set of values.

    While the set the distribution is defined over can be a set of arbitrary entities, each is given an 0-based index
    for representation in this class.
    """

    def __init__(self, pmf: np.ndarray) -> None:
        """
        Constructs the distribution object from the PMF.

        :param pmf: A vector of values containing the probability of each index.
        """
        super().__init__()
        self._pmf = pmf / pmf.sum()

    def pmf(self, val: Union[int, Tuple[int, ...]] = None, shape: Optional[Tuple[int, ...]] = None) -> Union[float,
                                                                                                             np.ndarray]:
        """
        Returns the probability mass associated with an index. Supports multi-coordinate indexing.

        :param val: The integer index to evaluate the PMF function at or a tuple of integers for multi-coordinate
            indexing.

        :param shape: The shape of the array for multi-coordinate indexing or None (default). When None, the index is
            just the linear index.

        :return: The value of the PMF at the index.

        :raises: ValueError
        """
        if val is None:
            if shape is None: shape = (self._pmf.size)
            return self._pmf.copy().reshape(shape)
        else:
            if shape is not None:
                val = np.ravel_multi_index(val, shape)

            if 0 <= val <= self._pmf.size:
                return self._pmf[val]
            else:
                raise ValueError('Input value does not correspond to an element of support (should be an integer '
                                 'between 0 and {})'.format(self.numel()))

    def numel(self) -> int:
        """
        Number of elements the random variable may take.

        :return: Returns the cardinality of the set the random variable is defined on.
        """
        return self._pmf.size

    def mean(self) -> float:
        return np.arange(self.numel()) @ self._pmf

    def cov(self) -> float:
        return (np.arange(self.numel()) ** 2) @ self._pmf - self.mean() ** 2

    def sample(self, n: int = 1) -> Union[int, np.ndarray]:
        rands = np.random.rand(n)
        cum = np.cumsum(self._pmf)

        if n == 1:
            return np.argmax(rands <= cum)
        else:
            ret = np.zeros(n)

            for i in range(n):
                ret[i] = np.argmax(rands[i] <= cum)

            return ret


class GaussianDist(ContinuousDist):
    def __init__(self, mean: np.ndarray, cov: np.ndarray) -> None:
        super().__init__()
        self._mean = mean.flatten()
        self._cov = cov.copy()

    def pdf(self, val: np.ndarray) -> float:
        return multivariate_normal(val, self._mean, self._cov)

    def dim(self) -> int:
        return self._mean.size

    def mean(self) -> np.ndarray:
        return self._mean

    def cov(self) -> np.ndarray:
        return self._cov

    def sample(self, n: int = 1) -> Union[float, np.ndarray]:
        if self.dim() == 1:
            return np.random.normal(self._mean[0], np.sqrt(self._cov[0]), n)
        else:
            return np.random.multivariate_normal(self._mean, self._cov, n).transpose()


def kl(a: Union[FiniteDist, GaussianDist], b: Union[FiniteDist, GaussianDist]) -> float:
    """
    Computes the KL divergence KL(a || b) between two distributions a and b.

    :param a: Either a finite or a Gaussian distribution.
    :param b: Either a a finite or a Gaussian distribution.
    :returns: The KL divergence between the two distributions as a float.
    :raises: TypeError
    """

    if type(a) is FiniteDist and type(b) is FiniteDist:
        return (a.pmf() * np.log(a.pmf() / b.pmf())).sum(axis=None)
    elif type(a) is GaussianDist and type(b) is GaussianDist:
        b_cov_inv = np.linalg.inv(b.cov())

        return (1 / 2) * (b_cov_inv @ a.cov()).trace() \
            + (b.mean() - a.mean()).transpose() @ b_cov_inv @ (b.mean() - a.mean()) \
            - a.dim() + np.log(np.linalg.det(b.cov()) / np.linalg.det(a.cov()))
    else:
        raise TypeError('Both distributions must be of the same type.')

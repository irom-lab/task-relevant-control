import numpy as np
import cvxpy as cvx
from typing import Union, Callable
from trcontrol.framework.prob import dists



def mi(x: np.array, channel: Callable[[Union[dists.FiniteDist, dists.GaussianDist]], Union[dists.FiniteDist, dists.GaussianDist]]) -> float:
    '''
    Computes the mutual information between random variables X, Y in the Markov Chain X -> Y where X, Y are either both
    finite or Gaussian random variables.
    '''
    y_x = y_given_x * x
    y = np.sum(y_x, axis=1)
    y.shape = (-1, 1)
    return np.sum(y_x * np.log(y_x / (y.reshape((-1, 1)) * x)), axis=None)


def kl(a: np.array, b: np.array) -> float:
    '''
    Computes the kl divergence between two distributions
    '''
    return np.sum(a * np.log(a / b), axis=None)


def monotonize(arr: np.array) -> np.array:
    '''
    Given a set of data points {(x_k, y_k)}, this function returns a sequence
    of the data that is monotone in both x and y. This sequence is constructed
    following the steps:

    1. Sort data by x values.
    2. Remove any data point with a y value that is not increasing.
    '''
    inds = np.argsort(kls[:, 0])
    biggest = 0
    ind = 0
    mono = np.zeros_like(kls)

    for i in range(inds.shape[0]):
        if arr[inds[i], 1] >= biggest:
            biggest = arr[inds[i], 1]
            mono[ind, :] = arr[inds[i], :]
            ind = ind + 1

    return mono


def cc(channel: np.array) -> (float, np.array):
    '''
    Computes the chanel capacity of a kernel.
    '''
    n, m = channel.shape

    x = cvx.Variable(n)
    y = channel @ x
    c = np.sum(channel * np.log(channel), axis=0)
    I = c * x + cvx.sum(cvx.entr(y))

    objective = cvx.Minimize(-I)
    constraints = [cvx.sum(x) == 1, x >= 0]
    problem = cvx.Problem(objective, constraints)
    problem.solve()

    if problem.status == 'optimal':
        return (problem.value, x.value)
    else:
        return (np.nan, np.nan)
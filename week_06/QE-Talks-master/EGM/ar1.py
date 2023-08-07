"""Discretizing AR(1) process using Rouwenhorst method."""

import numpy as np
from numba import njit


@njit
def rouwenhorst_matrix(n, p):
    """Compute the transition matrix for the Rouwenhorst method.

    Parameters
    ----------
    n: number of states.
    p: parameter of the transition matrix.

    Returns
    -------
    Π: transition matrix.
    """
    Π0 = np.array([[p, 1 - p], [1 - p, p]])

    for n in range(3, n + 1):
        Π = np.zeros((n, n))

        Π[:-1, :-1] += p * Π0
        Π[:-1, 1:] += (1 - p) * Π0
        Π[1:, :-1] += (1 - p) * Π0
        Π[1:, 1:] += p * Π0
        # Divide all but the top and bottom rows by two so that the elements in each row sum to one.
        Π[1:-1, :] /= 2
        Π0 = Π  # Update Π0 for next iteration.

    return Π


@njit
def stationary_markov(Π):
    """Compute the stationary distribution of a Markov chain.

    Parameters
    ----------
    Π: transition matrix.

    Returns
    -------
    π: stationary distribution.
    """
    n = Π.shape[0]
    π = np.linalg.solve(np.eye(n) - Π.T + np.ones((n, n)), np.ones(n))
    return π


@njit
def discretization_ar1(ρ, σ, n):
    """Discretize AR(1) process {z_t} to finite Markov chain {y_t} using Rouwenhorst method.

    Parameters
    ----------
    ρ: persistence parameter of {z_t}.
    σ: standard deviation of z_t.
    n: number of states.

    Returns
    -------
    y: discretized states of {z_t}.
    Π: transition matrix of {y_t}.
    π: stationary distribution of {y_t}.
    """
    p = (1 + ρ) / 2
    psi = (n - 1) ** 0.5 * σ
    y = np.linspace(-psi, psi, n)
    Π = rouwenhorst_matrix(n, p)
    π = stationary_markov(Π)

    return y, Π, π
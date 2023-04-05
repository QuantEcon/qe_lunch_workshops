"""
Functions for implementing the algorithms in McFadden(1989).
"""

import numpy as np
from numba import njit
from scipy import optimize, stats
from math import erf

# 1. dgp
@njit
def dgp(
    b, N, m, low=0.0, high=5.0, seed=1688,
):
    """generate data set of attributes and choices.
       return: 
       1. N * m * 2 matrix of attributes of alternatives.
       2. N * m matrix of choices.
       
       b: preference parameter for attribute 2.
       N: number of obs.
       m: number of alternatives.
       low, high: bounds of uniform distribution of attributes z.  
    """
    np.random.seed(seed)

    # (to use numba) no keyword arg here
    Z = np.random.uniform(low, high, (N, m, 2))  # attributes matrix

    # heterogenous preference
    u = np.random.normal(0, 1, (N, m))
    std = np.sum(Z ** 2, axis=2) ** 0.5
    v = std * u
    # homogenous preference
    alpha = np.array([1, b]).reshape(1, 1, 2)

    U = np.sum(Z * alpha, axis=2) + v  # N * m utility matrix
    # choice matrix
    Y = np.zeros((N, m))
    # numba doesn't allow multiple slicing
    for (i, j) in zip(range(N), np.asarray(np.argmax(U, axis=1), dtype=np.int64)):
        Y[i, j] = 1

    return Z, Y


# 2. moments function
@njit
def mom(y, z):
    """empirical moments for an observation. (just return observed y)
       
       y: m*1 observed (endogenous) choice vector (m is number of alternatives).
       z: m*2 matrix of (exogenous) attributes. 
    """
    return y.reshape(-1, 1)  # return a column vector


# 3. simulators
@njit
def freq_sim(v, z, b):
    """simulate choice vector of an agent.
       return 1*m 0-1 vector of choice (m is number of alternatives).

       v: m*1 (one-dim) vector of heterogenous preference.
       z: m*2 matrix of attributes.
       b: preference parameter for the second attribute.
    """
    m = z.shape[0]
    alpha = np.array([1, b]).reshape(1, 2)  # m*2 preference matrix
    utils = np.sum(z * alpha, axis=1) + v  # m*1 vector of utilities
    y = np.zeros(m)  # choice vector
    y[np.argmax(utils)] = 1.0
    return y.reshape(-1, 1)  # return m*1 (0-1) vector of choice


@njit
def imp_sim(r, z, b):
    """simulate choice vector of an agent based on importance function (exponential distribution).
       return 1*m 0-1 vector of simulated choice. (m is the number of alternatives.)

       r: m*(m-1) matrix of (positive) values drawn from exponential distribution. 
       z: m*2 matrix of attributes.
       b: preference parameter for attribute two.
    """
    m = z.shape[0]
    y = np.zeros(m)  # simulation results

    for j in range(m):  # loop for each alternative
        φ = np.exp(-np.sum(r))  # pdf of exponential distribution at v
        z_j = z[j]  # one-dim array
        z_others = np.concatenate((z[:j, :], z[j + 1 :, :]), axis=0)
        mean = (z_j[0] - z_others[:, 0]) + (z_j[1] - z_others[:, 1]) * b
        cov = np.sum(z_j ** 2) * np.ones((m - 1, m - 1)) + np.diag(
            np.sum(z_others ** 2, axis=1)
        )
        f = np.ravel(
            np.linalg.det(2 * np.pi * cov) ** (-0.5)
            * np.exp(
                -0.5
                * (
                    (r[j] - mean).reshape(1, -1)
                    @ np.linalg.inv(cov)
                    @ (r[j] - mean).reshape(-1, 1)
                )
            )
        )  # normal pdf at v
        y[j] = f[0] / φ

    return y.reshape(-1, 1)


@njit
def stern_sim(u, z, b):
    """simulate choice prob vector of an agent based on stern simulator (1992).
       return 1*m 0-1 vector of simulated choice. (m is the number of alternatives.)

       u: m*(m-1) matrix drawn from standard normal distribution. 
       z: m*2 matrix of attributes.
       b: preference parameter for attribute two.
    """
    m = z.shape[0]
    y = np.zeros(m)  # simulation results

    for j in range(m):  # loop for each alternative
        z_j = z[j]  # one-dim array
        u_j = u[j]
        z_others = np.concatenate((z[:j, :], z[j + 1 :, :]), axis=0)
        diff = (z_j[0] - z_others[:, 0]) + (z_j[1] - z_others[:, 1]) * b
        cov = np.sum(z_j ** 2) * np.ones((m - 1, m - 1)) + np.diag(
            np.sum(z_others ** 2, axis=1)
        )
        prob = 1.0  # initialize prob of choosing j
        λ = np.min(np.linalg.eig(cov)[0])  # minimal eigenvalue of cov matrix
        if λ > 0:
            a = λ ** 0.5
            e_values, e_vectors = np.linalg.eig(cov - λ * np.eye(m - 1))
            e_values[e_values < 0] = 0
            C = (
                e_vectors @ np.diag(np.sqrt(e_values)) @ e_vectors.T
            )  # square root of matrix

            for l in range(m - 1):  # loop for the other alternatives
                # to choose j, the difference between unobserved utility should be at least greater
                # than diff[l]
                prob *= 1 - 0.5 * (
                    1 + erf((-diff[l] - np.sum(C[l] * u_j)) / (a * 2 ** 0.5))
                )
        else:
            prob = 0

        y[j] = prob

    return y.reshape(-1, 1)


# 4. IV function
@njit
def simple_iv(z):
    """
    Generate a naive IV matrix (constant and z) for an observation.

    z: m*2 exogenous variable matrix of an observation. 
    """
    return np.concatenate((np.ones((1, z.shape[0])), z.T), axis=0)


# 4. estimator
@njit
def msm_criteria(
    b, Z, Y, iv, mom_func, simulator, V, S, W=None,
):
    """MSM criteria function.
    
       b: parameters of the model.
       Z: data matrix of exogenous variables. (first dim is obs)
       Y: data matrix of (observed) endogenous variables. (first dim is obs)
       iv: iv function of z.
       mom_func: moment function of (y, z).
       simulator: simulator of moments w.r.t. (v, z, b), where v is the drawn from its distribution.
       V: N*S*... matrix of v drawn from its distribution.(dim of V depends on the simulator)
       S: number of simulations.  
       W: weighting matrix. (identity matrix by default)
    """
    n = Z.shape[0]  # number of observations
    p = iv(Z[0]).shape[0]  # number of technical moment conditions (rows of IV matrix)
    q = mom_func(Y[0], Z[0]).shape[0]  # number of raw moments (rows of mom(yi, zi))

    if W == None:
        W = np.eye(p)

    g = np.zeros((p, 1))  # sum of moment conditions for all observations
    for i in range(n):
        sim_mom = np.zeros((q, 1))
        # simulated empirical moments
        for s in range(S):
            sim_mom += (1 / S) * simulator(V[i, s,], Z[i], b)
        g += iv(Z[i]) @ (mom_func(Y[i], Z[i]) - sim_mom)

    return (g.T @ W @ g)[0, 0]  # change scalar array to number


def msm_estimator(
    Z,
    Y,
    iv,
    mom_func,
    simulator,
    S,
    W=None,
    x0=0,
    seed=476,
    method="Nelder-Mead",
    simulator_name="frequency",
):
    """MSM estimator.

       Z: n*m*2 data matrix of exogenous variables. (first dim is obs)
       Y: n*m data matrix of (observed) endogenous variables. (first dim is obs)
       iv: iv function of z.
       mom_func: moment function of (y, z).
       simulator: simulator of moments w.r.t. (u, z, b), where u is the drawn from distribution f.
       W: weighting matrix. (identity matrix by default)
       S: number of simulations.  
       seed: used for simulation.   
       method: method used in scipy minimizer. (frequency simulator can only use Nelder-Mead)
       simulator_name: "frequency", "stern" (this determine how to draw random terms.)
    """
    np.random.seed(seed)
    N = Z.shape[0]  # number of observations
    m = Z.shape[1]  # number of alternatives

    # create random terms for simulation
    if simulator_name == "frequency":
        V = np.empty((N, S, m))
        u = np.random.normal(
            0, 1, (N, S, m)
        )  # use np.random.normal like this for numba!!
        std = np.sum(Z ** 2, axis=2) ** 0.5
        for s in range(S):
            V[:, s, :] = std * u[:, s, :]
    elif simulator_name == "stern":
        V = np.random.normal(0, 1, (N, S, m, m - 1))
    else:
        raise TypeError("Unknown Simulator!")

    # criteria function
    def f(b):
        # float(b)!! or python will treat b as array.
        # scale the criteria
        return msm_criteria(float(b), Z, Y, iv, mom_func, simulator, V, S, W,) / 1e6

    # BFGS doesn't work for frequency simulator
    # since criteria is not continuous under frequency simulator.
    res = optimize.minimize(f, x0=x0, method=method, tol=None)

    if res.success == False:
        print(res)
        raise ValueError("minimizer cannot be found!")

    # print(res)
    return res.x


@njit
def approx_moments(b, z, simulator, U):
    """use large number of simulations to approximate 
       the conditional theoretical moments for an observation.

       z: m*2 data matrix of exogenous variables. (first dim is obs)
       simulator: simulator of moments w.r.t. (u, z, b), where u is the drawn from normal distribution.
       (Notice that only stern simulator is allowed now!)

       U: S2 * m * m-1 matrix of random terms.
       (S2 is the number of simulations used to approximate the theoretical moments.)
       
       seed: the seed used for drawing u.
    """
    m = z.shape[0]  # number of alternatives
    y = np.zeros((m, 1))  # simulation results

    S2 = U.shape[0]
    for s in range(S2):
        y += (1 / S2) * simulator(U[s], z, b)

    return y


@njit
def cov_estimator(
    b,
    Z,
    Y,
    iv,
    mom_func,
    simulator,
    W=None,
    seed=2008,
    simulator_name="stern",
    S=1,
    S2=100,
    optimal_weighting=False,
):
    """estimate 1.the asymptotic covariance matrix for smooth simulator, 
                2. optimal weighting matrix.

       b: estimates.
       Z: n*m*2 data matrix of exogenous variables. (first dim is obs)
       Y: n*m data matrix of (observed) endogenous variables. (first dim is obs)
       iv: iv function of z.
       mom_func: moment function of (y, z).
       simulator: simulator of moments w.r.t. (u, z, b), where u is the drawn from distribution f.
       S: number of simulations.  
       W: weighting matrix. (identity matrix by default)
       seed: used for simulation.   
       simulator_name: "stern" (this determine how to simulate theoretical moments.)
       S2: how many draws are used to simulate the theoretical moments. (S2 should be large enough)
       optimal_weighting: whether to use the simplified covariance matrix for optimal weighting matrix.
    """
    assert simulator_name == "stern", "The simulator is not allowed!!"
    N = Z.shape[0]  # number of observations
    m = Z.shape[1]  # number of alternatives
    p = iv(Z[0]).shape[0]  # number of technical moment conditions (rows of IV matrix)
    if W == None:
        W = np.eye(p)

    np.random.seed(seed)
    U = np.random.normal(0, 1, (S2, m, m - 1))

    D = np.zeros((p, 1))  # gradient of moment conditions w.r.t b
    h = 1e-6  # diff used for calculate numerical gradient
    for i in range(N):
        D += (
            (1 / N)
            * iv(Z[i])
            @ (
                (
                    approx_moments(b + 0.5 * h, Z[i], simulator, U)
                    - approx_moments(b - 0.5 * h, Z[i], simulator, U)
                )
                / h
            )
        )

    # we need to return optimal weighting matrix
    # numba only accept one return statement
    optimal_W = np.zeros((p, p))

    if optimal_weighting == False:
        # construct sandwich form covariance matrix
        bread = np.linalg.inv(D.T @ W @ D) @ D.T @ W
        meat = np.zeros((p, p))
        U1 = np.random.normal(0, 1, (N, m, m - 1))  # random terms in simulators
        for i in range(N):
            thm_mom = approx_moments(
                b, Z[i], simulator, U
            )  # approximated theoretical moments conditional on Zi
            gmm_var = (
                iv(Z[i])
                @ (mom_func(Y[i], Z[i]) - thm_mom)
                @ (mom_func(Y[i], Z[i]) - thm_mom).T
                @ iv(Z[i]).T
            )
            sim_noise = (
                (1 / S)
                * iv(Z[i])
                @ (simulator(U1[i], Z[i], b) - thm_mom)
                @ (simulator(U1[i], Z[i], b) - thm_mom).T
                @ iv(Z[i]).T
            )
            meat += (1 / N) * (gmm_var + sim_noise)

        cov = bread @ meat @ bread.T
        optimal_W += np.linalg.inv(meat)

    else:
        cov = np.linalg.inv(D.T @ W @ D)

    return cov, optimal_W


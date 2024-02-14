"""
3D EGM algorithm for solving the consumption-savings-working-studying problem.
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from numba import njit
from scipy.interpolate import Rbf
from ar1 import discretization_ar1


# 1. EGM algorithm for solving the consumption-savings problem.


def egm3d_factory(
    β=0.98,
    r=0.1,
    e_min=0.1,
    e_max=10,
    n_e=10,
    q_min=0.1,
    q_max=10,
    n_q=10,
    a_min=0.1,
    a_max=10.0,
    n_a=20,
    ρ=0.975,
    σ=0.5,
    n_w=5,
    a1=0.3,
    a2=1,
    b1=0.3,
    b2=1,
    γ=0.5,
):
    """Factory of functions for the household problem.

    Parameters:
    -----------
    β: discount factor.
    r: net interest rate.
    e_min: minimum value of next period experience.
    e_max: maximum value of next period experience.
    n_e: number of points in next period experience grid.
    q_min: minimum value of next period knowledge.
    q_max: maximum value of next period knowledge.
    n_q: number of points in next period knowledge grid.
    a_min: minimum value of end-of-period wealth grid.
    a_max: maximum value of end-of-period wealth grid.
    n_a: number of points in end-of-period wealth grid.
    ρ: AR(1) coefficient for log income.
    σ: standard deviation of log income.
    n_w: number of points in income grid.

    Returns:
    --------
    egm: EGM solver for household's consumption policy function.
    draw_plot: a function which draws 2-d plots of the policy functions.
    """
    # discretize AR(1) process for income
    # log_y_grid: grid points for log income
    # Π_y: transition matrix for income
    # π_y: stationary distribution for income
    log_w_grid, Π_w, π_w = discretization_ar1(ρ, σ, n_w)
    w_grid = np.exp(log_w_grid)  # labor supply grid

    # next period experience grid
    e_grid = np.linspace(e_min, e_max, n_e)

    # next period knowledge grid
    q_grid = np.linspace(q_min, q_max, n_q)

    # end-of-period wealth grid
    a_grid = np.linspace(a_min, a_max, n_a)

    # number of wealth grids is same as end-of-period wealth grid
    n_m = n_a
    m_grid = np.linspace(a_min, a_max, n_m)  # wealth is always greater than 0

    # states grids
    X = np.array(
        [[w, e, q, m] for w in w_grid for e in e_grid for q in q_grid for m in m_grid]
    )

    # post decision states grids (reach row is one observation 1*4 vector)
    F = np.array(
        [
            [w_i, e, q, a]
            for w_i in range(n_w)
            for e in e_grid
            for q in q_grid
            for a in a_grid
        ]
    )

    # define marginal utility functions

    def u1_prime(h):
        """Marginal utility of working."""
        # return -2 * h
        return -np.exp(h)

    def u2_prime(l):
        """Marginal utility of studying."""
        return -np.exp(l)

    def u3_prime(c):
        """Marginal utility of consumption."""
        return 1 / c

    # define inverse marginal utility functions
    def u1_prime_inv(u):
        """Inverse of marginal utility of working."""
        return np.log(-u)

    def u2_prime_inv(u):
        """Inverse of marginal utility of studying."""
        return np.log(-u)

    def u3_prime_inv(u):
        """Inverse of marginal utility of consumption."""
        return 1 / u

    Φ1, Φ2, Φ3 = a1 / a2, b1 / b2, -1.0  #  Φ1, Φ2, Φ3

    def L1(h, l, c):
        return Φ1 * u1_prime(h) + (Φ2 - Φ1) * u2_prime(l) + (Φ3 - Φ2) * u3_prime(c)

    def L2(h, l, c):
        return Φ2 * u2_prime(l) + (Φ3 - Φ2) * u3_prime(c)

    def L3(h, l, c):
        return Φ3 * u3_prime(c)

    def g1(w, e, q):
        """partial derivative of M_{t+1} w.r.t. E_{t+1}."""
        return w * (1 - γ) * (q / e) ** γ

    def g2(w, e, q):
        """partial derivative of M_{t+1} w.r.t. Q_{t+1}."""
        return w * γ * (e / q) ** (1 - γ)

    def euler(w0_index, e1, q1, a0, policy):
        """get choice h0, l0, c0 given post decision states.
        policy: 2d grids represents a function of (e, q, m) -> (h, l, c),
        rows are observations, columns are choices (h, l, c).
        """
        rhs1, rhs2, rhs3 = 0.0, 0.0, 0.0

        for w1, π in zip(w_grid, Π_w[w0_index]):
            m1 = (1 + r) * a0 + w1 * (e1 ** (1 - γ)) * (q1**γ)
            h1, l1, c1 = policy(w1, e1, q1, m1)
            rhs1 += β * a2 * π * (L1(h1, l1, c1) + g1(w1, e1, q1) * L3(h1, l1, c1))
            rhs2 += β * b2 * π * (L2(h1, l1, c1) + g2(w1, e1, q1) * L3(h1, l1, c1))
            rhs3 += β * π * (-1) * (1 + r) * L3(h1, l1, c1)
        h0, l0, c0 = u1_prime_inv(rhs1), u2_prime_inv(rhs2), u3_prime_inv(rhs3)
        h0, l0, c0 = np.maximum(
            [u1_prime_inv(rhs1), u2_prime_inv(rhs2), u3_prime_inv(rhs3)],
            [0.0, 0.0, 0.0],
        )
        return h0, l0, c0

    def recover_states(h0, l0, c0, e1, q1, a0):
        """generates endogenous states given post decision states give policy function."""
        e0 = (e1 - a2 * h0) / a1
        q0 = (q1 - b2 * l0) / b1
        m0 = a0 + c0
        return e0, q0, m0

    def egm_step(policy0):
        """update policy functions"""
        X1 = np.empty_like(X)  # endogenous states
        choices = np.empty(
            (n_w * n_e * n_q * n_a, 3)
        )  # choices on the endogenous states
        for i, (w0_index, e1, q1, a0) in enumerate(F):  # loop over post decision states
            w0_index = int(w0_index)
            h0, l0, c0 = euler(w0_index, e1, q1, a0, policy0)
            e0, q0, m0 = recover_states(h0, l0, c0, e1, q1, a0)
            X1[i] = w_grid[w0_index], e0, q0, m0
            choices[i] = h0, l0, c0
        # update policy function (inaccurate interpolation for small number of points!!)
        policy1 = Rbf(*X1.T, choices, mode="N-D")

        return policy1

    def egm_3d(tol=1e-6, max_iter=10_000, details=False):
        """update policy functions"""
        policy0 = lambda w, e, q, m: (0.01 * e, 0.01 * q, 0.5 * m)
        for i in range(max_iter):
            print(i) if details else None
            policy1 = egm_step(policy0)
            if i >= 1:
                # we use consumption policy as convergence criteria
                diff = np.max(np.abs(policy1(*X.T) - policy0(*X.T)))
                if diff < tol:
                    return policy1
                else:
                    print(diff) if details else None
            policy0 = policy1
        else:
            print("No convergence.")

    def egm_3d_plot(policy, m_list=np.linspace(0.1, 10, 100)):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 4))
        ax1.set_xlabel("M")
        ax2.set_xlabel("M")
        ax3.set_xlabel("M")
        ax1.set_ylabel("c")
        ax2.set_ylabel("h")
        ax3.set_ylabel("l")
        for w, e, q in [(1.0, 1.0, 1.0), (1.0, 2.0, 2.0), (1.0, 5.0, 5.0)]:
            ax1.plot(
                m_list,
                [policy(w, e, q, m)[2] for m in m_list],
                label=f"w={w:.1f}, E={e:.1f}, Q={q:.1f}",
            )
            ax1.set_title("consumption policy")
        for w, e, q in [(1.0, 1.0, 1.0), (1.0, 2.0, 2.0), (1.0, 5.0, 5.0)]:
            ax2.plot(
                m_list,
                [policy(w, e, q, m)[0] for m in m_list],
                label=f"w={w:.1f}, E={e:.1f}, Q={q:.1f}",
            )
            ax2.set_title("working time policy")
        for w, e, q in [(1.0, 1.0, 1.0), (1.0, 2.0, 2.0), (1.0, 5.0, 5.0)]:
            ax3.plot(
                m_list,
                [policy(w, e, q, m)[1] for m in m_list],
                label=f"w={w:.1f}, E={e:.1f}, Q={q:.1f}",
            )
            ax3.set_title("learning time policy")
        ax1.legend()
        ax2.legend()
        ax3.legend()
        plt.show()

    return egm_3d, egm_3d_plot


if __name__ == "__main__":
    egm_3d, *_ = egm3d_factory(n_a=20, n_e=5, n_q=5, n_w=5, ρ=0.975, σ=0.4, γ=0.3)
    print(egm_3d(tol=1e-3, details=True))

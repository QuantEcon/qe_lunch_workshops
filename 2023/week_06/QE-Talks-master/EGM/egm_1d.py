"""
EGM algorithm for solving the consumption-savings problem.
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from numba import njit
from ar1 import discretization_ar1


def egm_factory(
    β=0.98,
    r=0.0025,
    a_min=0.0,
    a_max=20.0,
    n_a=100,
    ρ=0.975,
    σ=0.5,
    n_y=20,
):
    """Factory of functions for the household problem.

    Parameters:
    -----------
    β: discount factor.
    r: net interest rate.
    a_min: minimum value of end-of-period wealth grid.
    a_max: maximum value of end-of-period wealth grid.
    n_a: number of points in end-of-period wealth grid.
    ρ: AR(1) coefficient for log income.
    σ: standard deviation of log income.
    n_y: number of points in income grid.

    Returns:
    --------
    egm: EGM solver for household's consumption policy function.
    draw_plot: a function which draws 2-d and 3-d plots of the consumption policy function.
    """
    # discretize AR(1) process for income
    # log_y_grid: grid points for log income
    # Π_y: transition matrix for income
    # π_y: stationary distribution for income
    log_y_grid, Π_y, π_y = discretization_ar1(ρ, σ, n_y)
    y_grid = np.exp(log_y_grid)  # labor supply grid
    # end-of-period wealth grid
    a_grid = np.linspace(a_min, a_max, n_a)
    # number of wealth grids is same as end-of-period wealth grid
    n_m = n_a
    m_grid = np.linspace(0.01, a_max, n_m)  # wealth is always greater than 0

    @njit
    def u(c):
        """Utility function."""
        return np.log(c)

    @njit
    def u_prime(c):
        """Derivative of utility function."""
        return 1 / c

    @njit
    def u_prime_inv(u):
        """Inverse of derivative of utility function."""
        return 1 / u

    @njit
    def egm_step(p_c0):
        """Iterate policy function based on FOC and envelop condition.

        Parameters
        ----------
        p_c0 : initial consumption policy on (m, y) grids.

        Returns
        -------
        p_c1 : updated consumption policy on (m, y) grids.
        """
        # m' on (a, y) grids
        m1_grid = (1 + r) * a_grid.reshape(-1, 1) + y_grid.reshape(1, -1)

        # c(m', y) on (a, y) grids
        c1_grid = np.empty((n_a, n_y))
        for i in range(n_y):
            c1_grid[:, i] = np.interp(m1_grid[:, i], m_grid, p_c0[:, i])

        # E[u'(c(m', y))] on (a, y) grids
        ex_u_prime_grid = np.empty((n_a, n_y))
        for i in range(n_a):
            for j in range(n_y):
                ex_u_prime_grid[i, j] = np.dot(Π_y[j, :], u_prime(c1_grid[i, :]))

        # updated consumption policy on (a, y) grids
        c0_grid = u_prime_inv(β * (1 + r) * ex_u_prime_grid)

        # endogenous (m, y) grid
        m_y_grid = a_grid.reshape(-1, 1) + c0_grid

        # updated consumption policy on (m, y) grids
        p_c1 = np.empty((n_m, n_y))
        for i in range(n_y):
            # handle corner solution (borrowing constraint: consumption cannot exceed wealth)
            p_c1[:, i] = np.minimum(
                np.interp(m_grid, m_y_grid[:, i], c0_grid[:, i]), m_grid
            )

        return p_c1

    @njit
    def egm(max_iter=10_000_000, tol=1e-9, details=False):
        """Solve the consumption policy function using EGM.

        Parameters
        ----------
        tol: tolerance for convergence.
        max_iter: maximum number of iterations.

        Returns
        -------
        p_c: nz * na grids of consumption policy function.
        a_prime: nz * na grids of next period asset policy function.
        """
        # initial guess of consumption policy (consume 50% of wealth)
        p_c0 = 0.5 * m_grid.repeat(n_y).reshape(n_m, n_y)
        # iterate until convergence
        for i in range(max_iter):
            p_c1 = egm_step(p_c0)
            if np.max(np.abs(p_c1 - p_c0)) < tol:
                print(f"Converged in {i+1} iterations.") if details == True else None
                return p_c1
            else:
                p_c0 = p_c1
        else:
            raise ValueError("No converge.")

    def draw_policy(p_c, y_index_list, m_index_list):
        """Draw 3-D policy function, and 2-D consumption policy function for given income levels.
        p_c: consumption policy function on (m, y) grids.
        """

        # Set up a figure half as tall as it is wide
        fig = plt.figure(figsize=plt.figaspect(0.5))
        fig.suptitle("Consumption Policy Function")

        # 2-d m-c plot
        ax1 = fig.add_subplot(1, 3, 1)
        for i in y_index_list:
            y = y_grid[i]
            ax1.plot(m_grid, p_c[:, i], label=f"y={y:.2f}")
        ax1.legend()
        ax1.set_xlabel("M")
        ax1.set_ylabel("c")

        # 2-d y-c plot
        ax1 = fig.add_subplot(1, 3, 2)
        for i in m_index_list:
            m = m_grid[i]
            ax1.plot(y_grid, p_c[i, :], label=f"m={m:.2f}")
        ax1.legend()
        ax1.set_xlabel("y")
        ax1.set_ylabel("c")

        # 3-d plot
        ax3 = fig.add_subplot(1, 3, 3, projection="3d")
        M, Y = np.meshgrid(m_grid, y_grid)
        ax3.plot_surface(
            M,
            Y,
            p_c.T,
            rstride=1,
            cstride=1,
            linewidth=0,
            antialiased=False,
            cmap=mpl.cm.coolwarm,
        )
        ax3.set_xlabel("M")
        ax3.set_ylabel("Y")
        ax3.set_zlabel("c")

        plt.show()

    return egm, draw_policy


if __name__ == "__main__":
    egm, draw_policy = egm_factory(n_a=100, n_y=20, σ=0.7)
    p_c = egm(details=True)
    draw_policy(p_c, y_index_list=[0, 5, 9], m_index_list=[0, 10, 30])

Control variate

reference

- https://en.wikipedia.org/wiki/Control_variates
- https://www.math.nyu.edu/~goodman/teaching/MonteCarlo2005/notes/VarianceReduction.pdf

Here is n a warm afternoon, my supervisor John met with me 

In Science it is always a good habit to define the things you want to explain. Before we introduce an important question in Quantitative Economics please allow me to give the definitions of two very basic concepts in Statistics.

- a statistic
- expected value
- variance
- Unbiased estimator

Setups for an important question in Quantitative Economics:

- Let $\mu$ be a unknown parameter of interest.
- assume that we have a statistic $m$ such that its expected value is $\mu$, i.e., $\mathbb E [m] = \mu$
  - i.e., $m$ is an unbiased estimator for $\mu$.
- Let $\text{var} (m)$ be the variance of $m$.

Question

- So can we find an estimator with mean of $\mu$ but variance less than $\text{var} (m)$?
- and how?

The answer to the first question is YES!

Let me show you the magic HOW by answering the second one.

Suppose that we have another statistic $t$ such that $\mathbb E[t] = \tau$, which is a known value.

Construct 
$$
m^* = m + c ( t - \tau) \tag{1}
$$
which is an unbiased estimator for $\mu $ for any choice of the coefficient $c$.

Then the variance of the resulting estimator $m^*$ is 
$$
\text{ var } (m^*) = \text{ var } (m + c ( t - \tau))\\
= \text{ var } (m) + c^2 \text{ var } ( t - \tau)  +  2 c  \cdot \text{cov} (m, t) \\
= \text{ var } (m) + c^2 \text{ var } ( t )  +  2 c  \cdot \text{cov} (m, t) \tag{2}
$$

- the last equality is due to $\text{var}(\tau)=0$.

Differentiating Eq. (2) w.r.t. $c$ gives
$$
c^* = - \frac{\text{cov} (m, t)}{\text{ var } ( t )} \tag{3}
$$
 Plugging Eq. (5) into Eq. (4) yields the minimized variance of $m^*$
$$
\text{ var } (m^*) = \text{ var }(m) + c^2 \text{ var } ( t )  +  2 c  \cdot \text{cov} (m, t) \\
= \text{ var }(m) - \frac{[\text{cov} (m, t)]^2}{\text{ var } ( t )} \\
= ( 1 - \rho^2_{m, t}) \text{ var } (m) \tag{4}
$$

- where $\rho_{m, t} = \text{corr} (m, t)$ is the correlation coefficient of $m$ and $t$.

The greater the value of $|\rho_{m, t}|$, the greater the variance reduction can be achieved.
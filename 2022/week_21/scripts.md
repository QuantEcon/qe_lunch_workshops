motivation

- Monte carlo simulation
  - Let's consider an example of Monte Carlo
    - Let's see how Monte Carlo simulation works
      - Let $\mu$ be a unknown parameter of interest.
      - assume that we have a statistic $m$ such that its expected value is $\mu$, i.e., $\mathbb E [m] = \mu$
        - i.e., $m$ is an unbiased estimator for $\mu$.
      - Let $\text{var} (m)$ be the variance of $m$.
      - The Monte Carlo simulation can help us estimate the unknow parameter of interest if we know one of its unbiased estimator.
  - Question

    - We know if there were a class of unbiased estimators for this unkown parameter, then the variances of those unbiased estimators might be different.
    - A good statistc not only requires to be unbiased but also consider the consistency. 
      - the less variance it has the more consistent an unbiased estimator can be.
    - So in our monte carlo setup, if we were given an unbiased estimator for a unknown parameter of interest, then can we find another unbiased estimator with the less variance?

Control variate

- A simple answer to our question is Yes!

  - The control variate states that

    - Assume that $m$ is an unbiased estimator for the unknown but interested parameter $\mu$.

    - Given any other statistic $t$ with known expected value, i.e. $\mathbb E[t]= \tau$, we can estimate an optimal coefficient $c^*= - \frac{\text{cov} (m, t)}{\text{ var } ( t )}$ s.t. a newly constructed unbiased estimator
      $$
      m^* = m + c ( t - \tau) \tag{1}
      $$
      with 
      $$
      var (m^*) < var (m)
      $$
      

- Let's consider the example again.

- Let's think about the math behind

  - Let $m^*$ be the newly constructed statistic stated in (1).

  - Then the variance of the resulting estimator $m^*$ is 
    $$
    \text{ var } (m^*) = \text{ var } (m + c ( t - \tau))\\
    = \text{ var } (m) + c^2 \text{ var } ( t - \tau)  +  2 c  \cdot \text{cov} (m, t) \\
    = \text{ var } (m) + c^2 \text{ var } ( t )  +  2 c  \cdot \text{cov} (m, t) \tag{2}
    $$

    - the last equality is due to $\text{var}(\tau)=0$.

  - Differentiating Eq. (2) w.r.t. $c$ gives
    $$
    c^* = - \frac{\text{cov} (m, t)}{\text{ var } ( t )} \tag{3}
    $$

  - Plugging Eq. (5) into Eq. (4) yields the minimized variance of $m^*$
    $$
    \text{ var } (m^*) = \text{ var }(m) + c^2 \text{ var } ( t )  +  2 c  \cdot \text{cov} (m, t) \\
    = \text{ var }(m) - \frac{[\text{cov} (m, t)]^2}{\text{ var } ( t )} \\
    = ( 1 - \rho^2_{m, t}) \text{ var } (m) \tag{4}
    $$

    - where $\rho_{m, t} = \text{corr} (m, t)$ is the correlation coefficient of $m$ and $t$.

  - The greater the value of $|\rho_{m, t}|$, the greater the variance reduction can be achieved.
# Minimizing KL Divergence is Equivalent to Maxmimizing Likelihood

For a typical ML problem, all we have are samples from the true distribution $P(x)$ ie $data = \{(x_i)\}_{i=1}^N$ where $x_i \in \mathbb{R}^d$ are data points. We do not know the distribution $P(x)$ explicitly.

We try our best to estimate the true distribution $P(x)$ by $Q(x; \theta)$ where $\theta$ are the parameters of the model.

We want to know how well our model $Q(x; \theta)$ is performing. We can do this by calculating the KL divergence between the true distribution $P(x)$ and the estimated distribution $Q(x; \theta)$.

$D_{KL}(P || Q) = ∫ p(x) * log\left(\frac{p(x)}{q(x; \theta)}\right) dx$

$D_{KL}(P || Q) = E_{x \sim p(x)}[log\left(\frac{p(x)}{q(x; \theta)}\right)]$

$D_{KL}(P || Q) = E_{x \sim p(x)}[log\left(p(x)\right)] - E_{x \sim p(x)}[log\left(q(x; \theta)\right)]$

We are trying to find the parameters $\theta$ that minimize the KL divergence between $p(x)$ and $q(x; \theta)$.

hence $\theta^* = \underset{\theta}{argmin} \ D_{KL}(p || q(x; \theta))$

$\theta^* = \underset{\theta}{argmin} \ E_{x \sim p(x)}[log\left(p(x)\right)] - E_{x \sim p(x)}[log\left(q(x; \theta)\right)]$

because $E_{x \sim p(x)}[log\left(p(x)\right)]$ is constant with respect to $\theta$, we can ignore it.

$\theta^* = \underset{\theta}{argmax} \ E_{x \sim p(x)}[log\left(q(x; \theta)\right)]$

$E_{x \sim p(x)}[log\left(q(x; \theta)\right)]$ is called the **Expected Log Likelihood**,

By the law of large numbers, we can approximate the expected log likelihood by the average log likelihood of the data:

$E_{x \sim p(x)}[log(q(x; \theta))] \approx \frac{1}{N} \sum_{i=1}^N log(q(x_i; \theta))$

Therefore, our optimization problem becomes:

$\theta^* = \underset{\theta}{argmax} \ \frac{1}{N} \sum_{i=1}^N log(q(x_i; \theta))$

This is equivalent to maximizing the log likelihood of the data.

$\theta^* = \underset{\theta}{argmax} \ \frac{1}{N} \sum_{i=1}^N log(q(x_i; \theta))$

hence $\theta$ is also called the **maximum log likelihood estimate(MLE)**. 

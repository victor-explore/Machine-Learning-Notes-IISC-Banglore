# Maximum aposteriori estimate(MAP  )

The Maximum A Posteriori (MAP) estimate is defined as:

$$\theta^*_{MAP} = \arg\max_{\theta} p(\theta|V) = \arg\max_{\theta} p(V|\theta) p(\theta)$$

where
- $\theta^*_{MAP}$ is the MAP estimate of the parameter $\theta$
- $p(\theta|V)$ is the posterior probability of the parameter $\theta$ given the observed data $V$
- $p(V|\theta)$ is the likelihood of the observed data $V$ given the parameter $\theta$
- $p(\theta)$ is the prior probability of the parameter $\theta$

Note that:
- Unlike MLE, MAP estimation incorporates prior knowledge about the parameter $\theta$ and observed data $V$ where as MLE only depends on the observed data $V$.
- MAP provides a balance between the likelihood of the data and the prior beliefs, leading to more robust estimates, especially when the data is limited.

## Conjugate prior

A conjugate prior is a type of prior distribution $p(\theta)$ such that when multiplied with the likelihood function $p(V|\theta)$ the posterior distribution $p(\theta|V)$ that we get is in the same form as the prior $p(\theta)$.

In simple terms, the posterior $p(\theta)$ belongs to the same family of distributions as the prior $p(\theta|V)$.



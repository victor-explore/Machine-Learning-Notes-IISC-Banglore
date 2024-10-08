# Expectation Maximization (EM) algorithm

Let the data be $D = \{t_i\}_{i=1}^N$ where $t_i \in \mathbb{R}^d$ are iid $p_t(t)$ data points.

## Latent variable models

Let each data point $x_i$ be associated with a latent variable $z_i$ that is not observed. $z_i$ is a random variable that takes values in some finite set ${1, \ldots, K}$ and represents the membership of $x_i$ to one of $K$ clusters.

Hence the data can be represented as $D = \{(t_i, z_i)\}_{i=1}^N$ where $t_i$ is the observed data and $z_i$ is the latent variable and $(t_i,z_i)$ are iid $p_{tz}$.

The marginal distribution of the observed data is given by:

$$
p_t(t) = \sum_z p_{tz}(t, z)
$$

## Variational inference
We want to maximize the log-likelihood of the observed data:


$$
\begin{aligned}
\ell(\theta) &= \log \sum_z p_{tz}^\theta(t, z) \\
\end{aligned}
$$
Let $q(z)$ be an arbitrary distribution over $z$.
$$
\begin{aligned}
\ell(\theta) &= \log \sum_z q(z) \frac{p_{tz}^\theta(t, z)}{q(z)} \\
\end{aligned}
$$

$$
\ell(\theta) = \log \mathbb{E}_{q(z)} \left[ \frac{p_{tz}^\theta(t, z)}{q(z)} \right]
$$

By Jensen's inequality, we have:

$$
\log \mathbb{E}_{q(z)} \left[ \frac{p_{tz}^\theta(t, z)}{q(z)} \right] \geq \mathbb{E}_{q(z)} \left[ \log \frac{p_{tz}^\theta(t, z)}{q(z)} \right]
$$

Hence, we have:

$$
\ell(\theta) \geq \mathbb{E}_{q(z)} \left[ \log \frac{p_{tz}^\theta(t, z)}{q(z)} \right]
$$

$\ell(\theta)$ is the log-likelihood of the observed data it is also called the evidence.

$\mathbb{E}_{q(z)} \left[ \log \frac{p_{tz}^\theta(t, z)}{q(z)} \right]$ is the lower bound of the log-likelihood and is also called the **evidence lower bound (ELBO)**.

Note that the ELBO is a function of $q(z)$ and $\theta$.

## Optimizing ELBO
Instead of maximizing the evidence, we maximize the evidence lower bound (ELBO). We do it by maximizing the ELBO with respect to $q(z)$ and $\theta$.

## Making ELBO tight

To make the ELBO tight, we consider the difference between the evidence and the ELBO:

$$
\begin{aligned}
\ell(\theta) - \text{ELBO}(q, \theta) &= \log p_t^\theta - \mathbb{E}_{q(z)} \left[ \log \frac{p_{tz}^\theta(t, z)}{q(z)} \right] \\
\end{aligned}
$$

$$
\begin{aligned}
\ell(\theta) - \text{ELBO}(q, \theta) &= \log p_t^\theta - \mathbb{E}_{q(z)} \left[ \log \frac{p_{z|t}^\theta(z|t)p_t^\theta(t)}{q(z)} \right] \\
\end{aligned}
$$

$$
\begin{aligned}
\ell(\theta) - \text{ELBO}(q, \theta) &= \log p_t^\theta - \mathbb{E}_{q(z)} \left[ \log p_{z|t}^\theta(z|t) + \log p_t^\theta(t) - \log q(z) \right] \\
\end{aligned}
$$


$$
\begin{aligned}
\ell(\theta) - \text{ELBO}(q, \theta) &= \log p_t^\theta - \mathbb{E}_{q(z)} \left[ \log p_{z|t}^\theta(z|t) \right] - \mathbb{E}_{q(z)} \left[ \log p_t^\theta(t) \right] + \mathbb{E}_{q(z)} \left[ \log q(z) \right] \\
\ell(\theta) - \text{ELBO}(q, \theta) &= \log p_t^\theta - \mathbb{E}_{q(z)} \left[ \log p_{z|t}^\theta(z|t) \right] - \log p_t^\theta(t) + \mathbb{E}_{q(z)} \left[ \log q(z) \right] \\
\ell(\theta) - \text{ELBO}(q, \theta) &= -\mathbb{E}_{q(z)} \left[ \log p_{z|t}^\theta(z|t) \right] + \mathbb{E}_{q(z)} \left[ \log q(z) \right] \\
\ell(\theta) - \text{ELBO}(q, \theta) &= \mathbb{E}_{q(z)} \left[ \log q(z) - \log p_{z|t}^\theta(z|t) \right] \\
\ell(\theta) - \text{ELBO}(q, \theta) &= \mathbb{E}_{q(z)} \left[ \log \frac{q(z)}{p_{z|t}^\theta(z|t)} \right] \\
\ell(\theta) - \text{ELBO}(q, \theta) &= KL(q(z) || p_{z|t}^\theta(z|t))
\end{aligned}
$$


The difference between the evidence and the ELBO is the Kullback-Leibler (KL) divergence between $q(z)$ and $p_{z|t}^\theta(z|t)$. To make the ELBO tight, we need to minimize this KL divergence.

The KL divergence is always non-negative and equals zero if and only if the two distributions are identical. Therefore, to make the ELBO tight, we set:

$$
q(z) = p_{z|t}^\theta(z|t)
$$

This choice of $q(z)$ makes the ELBO equal to the evidence, achieving the tightest possible bound.

## EM Algorithm

The Expectation-Maximization (EM) algorithm is an iterative method to find maximum likelihood estimates of parameters in statistical models with latent variables. It consists of two main steps:

1. **E-step (Expectation)**: Compute the expected value of the log-likelihood function with respect to the conditional distribution of $z$ given $t$ under the current estimate of the parameters $\theta$.

2. **M-step (Maximization)**: Find the parameter that maximizes this expected log-likelihood.

Formally, the EM algorithm can be described as follows:

1. Initialize $\theta^{(0)}$
2. Repeat until convergence:
   - E-step: Compute $q^{(t)}(z) = p_{z|t}^{\theta^{(t-1)}}(z|t)$
   - M-step: 
     1. $\theta^{(t)} = \arg\max_\theta \mathbb{E}_{q^{(t)}(z)} \left[ \log \frac{p_{tz}^\theta(t, z)}{q(z)} \right]$
     2. $\theta^{(t)} = \arg\max_\theta \mathbb{E}_{q^{(t)}(z)} \left[ \log p_{tz}^\theta(t,z) \right] - \mathbb{E}_{q^{(t)}(z)} \left[ \log q(z) \right]$
     3. $\theta^{(t)} = \arg\max_\theta \mathbb{E}_{q^{(t)}(z)} \left[ \log p_{tz}^\theta(t,z) \right]$ as second term is constant wrt $\theta$

The EM algorithm guarantees that the likelihood increases at each iteration and converges to a local maximum.


## EM algorithm for GMM

Let's apply the EM algorithm to the Gaussian Mixture Model (GMM) we discussed earlier. Recall that in a GMM, we have:

- Observed data points: $\mathbf{x} = (x_1, ..., x_N)$
- Latent variables: $\mathbf{z}$, where $z_i \in \{1, ..., m\}$ indicates gaussian component
- $p_t(t) = \sum_{j=1}^m \alpha_j \mathcal{N}(t; \mu_j, \xi_j)$
- $\alpha_j$ are mixing coefficients, $\sum_{j=1}^m \alpha_j = 1$
- $\mu_j$ are mean vectors
- $\xi_j$ are covariance matrices
- $p(t_i | z_i = k) = \mathcal{N}(\mathbf{t_i}; \mu_k, \Sigma_k)$
- Parameters: $\theta = (\alpha_1, ..., \alpha_m, \mu_1, ..., \mu_m, \xi_1, ..., \xi_m)$

The EM algorithm for GMM proceeds as follows:

1. **Initialization**: 
   Choose initial values for the parameters $\theta = (\alpha_1, ..., \alpha_m, \mu_1, ..., \mu_m, \xi_1, ..., \xi_m)$.

2. **E-step**: 
   Compute the posterior probabilities (responsibilities) for each data point and each Gaussian component:

   $$
   q^{t+1}(z=s) = p_{z|t}^{\theta^k}(z=s|t=t_i) = \frac{p_{tz}^{\theta^k}(z)}{p_t^{\theta^k}(t)} = \frac{N(t;t_i,\mu_s,\xi_s)\alpha_s}{\sum_{j=1}^m \alpha_j N(t_j;\mu_j,\xi_j)}
   $$

3. **M-step**: 
   Update the parameters:

   $$
   \theta^{k+1} = \arg\max_\theta ELBO(q, \theta)
   $$
   $$
   \theta^{k+1} = \arg\max_\theta


   $$
   = \arg\max_\theta (E_q \log N(t,\mu_\xi,\xi)\alpha_s)
   $$

The EM algorithm for GMM alternates between these steps until convergence, effectively maximizing the likelihood of the observed data under the Gaussian mixture model.






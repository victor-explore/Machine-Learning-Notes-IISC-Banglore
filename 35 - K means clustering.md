# K-means clustering also known as K-means algorithm or hard clustering

## Recall that in mixture model clustering

Data was: $D = \{x_i\}_{i=1}^N$. We assumed latent variable $z_i$ is associated with each data point $x_i$. Hence data became $D = \{(x_i, z_i)\}_{i=1}^N$.

We assumed 
  
  $$p_\theta(x) = \sum_{j=1}^m p_\theta(x|z=j) p_\theta(z=j) = \sum_{j=1}^m N(x; \mu_j, \Sigma_j) \pi_j$$

Where:
- $N(x; \mu_j, \Sigma_j)$ is the Gaussian probability density function for cluster $j$, with mean $\mu_j$ and covariance matrix $\Sigma_j$.
- $\pi_j$ is the probability of a data point belonging to cluster $j$.

## K-means clustering

Now let's make the following assumptions to simplify our model:

- All clusters have the same variance $\sigma^2$.
- Each cluster is spherical, symmetric, and equally shaped ie $\Sigma_j = \sigma^2 I$ and $\sigma^2 \to 0$ where $I$ is the identity matrix.
- All clusters have equal prior probability, i.e., $\pi_j = \frac{1}{m}$ for all $j$.

With these assumptions, our model simplifies to:

$$p_\theta(x) = \sum_{j=1}^m \frac{1}{m} N(x; \mu_j, \sigma^2 I)$$

As $\sigma^2 \to 0$, the Gaussian distributions become increasingly peaked, and in the limit, they become delta functions centered at the means $\mu_j$. This leads to a hard assignment of each point to the nearest cluster center:

$$z_i = \arg\min_j ||x_i - \mu_j||^2$$

This is exactly what K-means does: it assigns each point to the nearest cluster center.

## K-means algorithm


1. Initialize: Randomly choose $m$ points as initial cluster centers $\{\mu_1, ..., \mu_m\}$.
2. Assign: For each data point $x_i$, assign it to the nearest cluster:
   $$z_i = \arg\min_j ||x_i - \mu_j||^2$$
3. Update: Recalculate the cluster centers as the mean of all points assigned to that cluster:
   $$\mu_j = \frac{1}{|C_j|} \sum_{i: z_i=j} x_i$$
   where $C_j$ is the set of all points assigned to cluster $j$.
4. Repeat steps 2 and 3 until convergence (i.e., when cluster assignments no longer change significantly).

Note that it can be shown that the above algorithm is equivalent to EM algorithm:
- E step is computing the new mean with existing cluster assignments.
- M step is reassigning data points to clusters based on the new mean.


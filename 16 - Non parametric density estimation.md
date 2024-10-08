# Non parametric density estimation

## What is parametric density estimation?
In parametric density estimation, we assume that the data is generated from a known distribution, such as the normal distribution, and we estimate the parameters of the distribution using various methods like maximum likelihood estimation or risk minimization.

## What is non-parametric density estimation?
However, in non-parametric density estimation, we make no assumptions about the form of the distribution and estimate the density directly from the data.

The basic idea behind non-parametric density estimation is to estimate the probability density function (PDF) directly from the data without assuming a specific functional form. One way to approach this is by considering the probability of a data point falling within a certain region.

Let $D = \{x_1, x_2, ..., x_n\}$ be our dataset. The probability of a data point falling within a region $R$ can be estimated as:

$$P(\text{data point in region } R) = \frac{\text{number of data points in region } R}{\text{total number of data points}} = \frac{k}{n}$$

where $k$ is the number of data points in region $R$, and $n$ is the total number of data points.

Also 

$$P(\text{data point in region } R) = p(x)* V $$

where $p(x)$ is the probability density at point $x$ and $V$ is the volume of the region $R$.

Combining the above two equations, we get:

$$\frac{k}{n} = p(x) * V$$

Rearranging this equation, we get:

$$p(x) = \frac{k}{nV}$$

This forms the basis for various non-parametric density estimation techniques.



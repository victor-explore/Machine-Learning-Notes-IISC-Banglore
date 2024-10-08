# Parzen window estimate also known as the kernel density estimate

## Basic idea

Recall that we had derived the following equation for non-parametric density estimation:

$$p(x) = \frac{k}{nV}$$

where:
- $k$ is the number of data points in region $R$
- $n$ is the total number of data points in the dataset $D$
- $V$ is the volume of the region $R$

In Parzen window estimate, we fix the volume $V$ and count $k$ by using a window function.

## Problem setting
Given a set of data points $D = \{x_1, x_2, ..., x_n\}$, we want to estimate the probability density function $p(x)$ at a given point $x$ ie model the distribution of data points in the dataset.

## Formulation using uniform kernel (rectangular kernel)

Let's define the formulation for the Parzen window estimate:

1. Define the volume $V_n$:
   $$V_n = (h_n)^d$$
   where:
   - $h_n$ is the length of the hypercube in $\mathbb{R}^d$
   - $d$ is the dimension of the data
   - $V_n$ is the volume of the hypercube in $\mathbb{R}^d$

2. Define the window function $\phi(u)$:
   $$\phi(u) = \begin{cases}
   1 & \text{if } |u_j| \leq \frac{1}{2}, j=1,\ldots,d \\
   0 & \text{otherwise}
   \end{cases}$$
   where:
   - $\phi(u)$ returns 1 if the point $u$ is within the unit hypercube centered at the origin, and 0 otherwise.
   - $u_j$ is the $j$th coordinate of the point $u$.

3. Then the window function centered at a data point $x_i$ is:
   $$\phi(\frac{x-x_i}{h_n}) = \begin{cases}
   1 & \text{if } x \text{ is in the hypercube centered at } x_i \text{ of side } h_n \\
   0 & \text{otherwise}
   \end{cases}$$

4. Count $k_n$, the number of points in the hypercube centered at $x_i$ of side $h_n$:
   $$k_n = \sum_{i=1}^n \phi(\frac{x-x_i}{h_n})$$

5. The Parzen window estimate:
   $$p(x) = \frac{k}{nV} = \frac{\sum_{i=1}^n \phi(\frac{x-x_i}{h_n})}{n(h_n)^d}$$

Note that:
- Here, $h_n$ is a hyperparameter that controls the width of the window.
- As $n \to \infty$, if $h_n \to 0$ and $nh_n^d \to \infty$, then the estimate converges to the true density.

## Parzen window with Gaussian kernel

1. Gaussian kernel:
   $$\phi(u) = \frac{1}{(2\pi)^{d/2}} e^{-\frac{1}{2} u^2}$$

2. The window function centered at a data point $x_i$ is:
   $$\phi(\frac{x-x_i}{h_n}) = \frac{1}{(2\pi)^{d/2}} e^{-\frac{1}{2} (\frac{x-x_i}{h_n})^2}$$

3. The Parzen window estimate:
   $$p(x) = \frac{k}{nV} = \frac{\sum_{i=1}^n \phi(\frac{x-x_i}{h_n})}{n(h_n)^d}$$

## Algorithm

1. Choose a value for $h_n$
2. For each test data point $x_i$, find the number of points in the hypercube centered at $x_i$ of side $h_n$
3. Calculate the Parzen window estimate $p(x)$ using the number of points found in the previous step



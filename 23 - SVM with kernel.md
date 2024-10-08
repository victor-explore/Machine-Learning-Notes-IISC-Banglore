# SVM with kernel

## Prerequisites - Kernel trick
Kernel function $k(x, y) = \phi(x)^T \phi(y)$ where $\phi(x)$ is a feature mapping that maps the data points to a higher dimensional space without actually computing the feature mapping, i.e., $\phi(x)$.

In other words, we can compute the kernel function $k(x, y)$ without actually computing the feature mapping $\phi(x)$.

Examples of kernel functions:
- Polynomial kernel: $k(x_1, x_2) = \phi(x_1)^T \phi(x_2) = (1 + x_1^T x_2)^p$
- Sigmoid kernel: $k_s(x_1, x_2) = \frac{1}{1 + \exp(a x_1^T x_2)}$
- RBF/Gaussian kernel: $k(x_1, x_2) = \exp\left(-\frac{\| x_1 - x_2 \|^2}{\sigma^2}\right)$

## Motivation
If the dataset $D = \{(x_i, y_i)\}_{i=1}^n$ where $x_i \in \mathbb{R}^d$ and $y_i \in \{-1, 1\}$ is not linearly separable in the original $d$-dimensional space, we can use a feature mapping $\phi(x)$ to map the data points to a higher dimensional space where they become linearly separable and then use the SVM optimization problem to find the optimal hyperplane.

Hence the new dataset $D' = \{(x_i', y_i)\}_{i=1}^n$ where $x_i' = \phi(x_i)$ and $y_i \in \{-1, 1\}$.

## Optimization problem
The optimization problem for SVM with kernel can be formulated as:

$$\min_{w,b,\xi} \frac{1}{2} \|w\|^2 + C \sum_{i=1}^n \xi_i$$

subject to:
$$y_i(w^T \phi(x_i) + b) \geq 1 - \xi_i, \quad i = 1,2,3,...,n$$
$$\xi_i \geq 0$$

## Solution to the optimization problem

The Lagrangian for the SVM optimization problem with kernel can be formulated as follows:

$$ L(w, b, \xi, \alpha, \beta) = \frac{1}{2} \| w \|^2 + C \sum_{i=1}^n \xi_i - \sum_{i=1}^n \alpha_i \left( y_i (w^T \phi(x_i) + b) - 1 + \xi_i \right) - \sum_{i=1}^n \beta_i \xi_i $$

where $\alpha_i \geq 0$ and $\beta_i \geq 0$ are the Lagrange multipliers.

To find the optimal solution, we take the partial derivatives of the Lagrangian with respect to $w$, $b$, $\xi_i$, $\alpha_i$, and $\beta_i$, and set them to zero:

$$ \frac{\partial L}{\partial w} = 0 \implies w = \sum_{i=1}^n \alpha_i y_i \phi(x_i) $$
$$ \frac{\partial L}{\partial b} = 0 \implies \sum_{i=1}^n \alpha_i y_i = 0 $$

$$ \frac{\partial L}{\partial \xi_i} = 0 \implies C - \alpha_i - \beta_i = 0 $$

$$ \frac{\partial L}{\partial \alpha_i} = 0 \implies y_i (w^T \phi(x_i) + b) - 1 + \xi_i \geq 0 $$

$$ \frac{\partial L}{\partial \beta_i} = 0 \implies \xi_i \geq 0 $$

The complementary slackness conditions are:

$$ \alpha_i (y_i (w^T \phi(x_i) + b) - 1 + \xi_i) = 0 $$
$$ \beta_i \xi_i = 0 $$

From these conditions, we can deduce:

1. If $0 < \alpha_i < C$, then $\xi_i = 0$ and $y_i (w^T \phi(x_i) + b) = 1$
2. If $\alpha_i = 0$, then $y_i (w^T \phi(x_i) + b) \geq 1$
3. If $\alpha_i = C$, then $y_i (w^T \phi(x_i) + b) \leq 1$

## Dual Formulation

Substituting these back into the Lagrangian and simplifying, we get the dual formulation:

$$ \max_{\alpha} \sum_{i=1}^n \alpha_i - \frac{1}{2} \sum_{i=1}^n \sum_{j=1}^n \alpha_i \alpha_j y_i y_j k(x_i, x_j) $$

subject to:

$$ \sum_{i=1}^n \alpha_i y_i = 0 $$
$$ 0 \leq \alpha_i \leq C \quad \text{for all } i $$

where $k(x_i, x_j) = \phi(x_i)^T \phi(x_j)$ is the kernel function.

## Finding $w$

Once we solve the dual problem and obtain the optimal $\alpha_i$, we can find $w$ and $b$:

$$ w = \sum_{i=1}^n \alpha_i y_i \phi(x_i) $$

but we don't need to compute $\phi(x_i)$ explicitly. Hence we can use the kernel function $k(x_i, x)$ to compute the decision function:

$$ w^T \phi(x) = \left(\sum_{i=1}^n \alpha_i y_i \phi(x_i)\right)^T \phi(x) = \sum_{i=1}^n \alpha_i y_i \phi(x_i)^T \phi(x) = \sum_{i=1}^n \alpha_i y_i k(x_i, x) $$

This is often referred to as the "kernel trick", which allows us to work in high-dimensional feature spaces without explicitly computing the feature vectors.

## Finding $b$

To find $b$, we can use any support vector (a point where $0 < \alpha_i < C$) and the fact that for these points, $y_i(w^T \phi(x_i) + b) = 1$.

## Decision Function

The decision function for classifying new points becomes:

$$ f(x) = \text{sign}\left(\sum_{i=1}^n \alpha_i y_i k(x_i, x) + b\right) $$

where only the support vectors (points with $\alpha_i > 0$) contribute to the sum.

The final classifier is:

$$ f(x) = \begin{cases}
  1 & \text{if } f(x) > 0 \\
 -1 & \text{if } f(x) < 0
\end{cases} $$

The main difference from the non-kernel SVM is the use of the kernel function $k(x_i, x)$ instead of the dot product $x_i^T x$. This allows the SVM to find non-linear decision boundaries in the original input space by implicitly working in a higher-dimensional feature space.




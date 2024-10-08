# Support vector machine(SVM) when data is not linearly separable

## Dataset is not linearly separable
Unlike the case when the dataset is linearly separable, we cannot find a hyperplane that separates the two classes of data points with zero training error.

## Soft margin

To handle this case, we introduce a slack variable $\xi_i$ for each data point $x_i$ to allow some points to be on the wrong side of the margin or even in the wrong class.

## Optimization problem

The optimization problem becomes:

$$ \min_{w, b, \xi} \frac{1}{2} \| w \|^2 + C \sum_{i=1}^n \xi_i $$
$$ \text{s.t. } y_i (w^T x_i + b) \geq 1 - \xi_i \text{ for all } i $$
$$ \xi_i \geq 0 \text{ for all } i $$



## Solution to the optimization problem

The Lagrangian for the SVM optimization problem can be formulated as follows:

$$ L(w, b, \xi, \mu, \nu) = \frac{1}{2} \| w \|^2 + C \sum_{i=1}^n \xi_i - \sum_{i=1}^n \mu_i \left( y_i (w^T x_i + b) - 1 + \xi_i \right) - \sum_{i=1}^n \nu_i \xi_i $$

where $\mu_i \geq 0$ and $\nu_i \geq 0$ are the Lagrange multipliers.

To find the optimal solution, we take the partial derivatives of the Lagrangian with respect to $w$, $b$, $\xi_i$, $\mu_i$, and $\nu_i$, and set them to zero:

$$ \frac{\partial L}{\partial w} = 0 \implies w = \sum_{i=1}^n \mu_i y_i x_i $$
$$ \frac{\partial L}{\partial b} = 0 \implies \sum_{i=1}^n \mu_i y_i = 0 $$

$$ \frac{\partial L}{\partial \xi_i} = 0 \implies C - \mu_i - \nu_i = 0 $$

$$ \frac{\partial L}{\partial \mu_i} = 0 \implies y_i (w^T x_i + b) - 1 + \xi_i \geq 0 $$

$$ \frac{\partial L}{\partial \nu_i} = 0 \implies \xi_i \geq 0 $$

The complementary slackness conditions are:

$$ \mu_i (y_i (w^T x_i + b) - 1 + \xi_i) = 0 $$
$$ \nu_i \xi_i = 0 $$

From these conditions, we can deduce:

1. If $0 < \mu_i < C$, then $\xi_i = 0$ and $y_i (w^T x_i + b) = 1$
2. If $\mu_i = 0$, then $y_i (w^T x_i + b) \geq 1$
3. If $\mu_i = C$, then $y_i (w^T x_i + b) \leq 1$

## Dual Formulation

Substituting these back into the Lagrangian and simplifying, we get the dual formulation:

$$ \max_{\mu} \sum_{i=1}^n \mu_i - \frac{1}{2} \sum_{i=1}^n \sum_{j=1}^n \mu_i \mu_j y_i y_j (x_i^T x_j) $$

subject to:

$$ \sum_{i=1}^n \mu_i y_i = 0 $$
$$ 0 \leq \mu_i \leq C \quad \text{for all } i $$

## Finding $w$ and $b$

Once we solve the dual problem and obtain the optimal $\mu_i$, we can find $w$ and $b$:

$$ w = \sum_{i=1}^n \mu_i y_i x_i $$

To find $b$, we can use any support vector (a point where $0 < \mu_i < C$) and the fact that for these points, $y_i(w^T x_i + b) = 1$.

## Decision Function

The decision function for classifying new points remains the same as in the linearly separable case:

$$ f(x) = \text{sign}\left(\sum_{i=1}^n \mu_i y_i (x_i^T x) + b\right) $$

where only the support vectors (points with $\mu_i > 0$) contribute to the sum.

The main difference from the linearly separable case is the upper bound $C$ on the Lagrange multipliers $\mu_i$, which allows for some misclassifications in the training set while still finding the optimal separating hyperplane.




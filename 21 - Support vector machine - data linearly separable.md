# Support vector machine(SVM) when data is linearly separable

## Dataset is linearly separable
Dataset $D = \{(x_i, y_i)\}_{i=1}^n$ where $x_i \in \mathbb{R}^d$ and $y_i \in \{-1, 1\}$ is said to be linearly separable if there exists a hyperplane that can separate the two classes of data points with zero training error.

Mathematically, a dataset is linearly separable if there exist weights $w$ and bias $b$ such that:
$$ w^T x_i + b > 1 \text{ for } y_i = 1 $$
$$ w^T x_i + b < -1 \text{ for } y_i = -1 $$

This can be rewritten as:
$$ y_i (w^T x_i + b) \geq 1 \text{ for all } i $$

In other words, there does not exist any $x_i$ such that $-1 \leq w^T x_i + b \leq 1$.

The distance between the margins is $\frac{2}{\| w \|}$.

## Optimization problem

Hence the SVM optimization problem is:
$$ \min_{w, b} \frac{1}{2} \| w \|^2 $$
$$ \text{s.t. } y_i (w^T x_i + b) \geq 1 \text{ for all } i $$

## Solution to the optimization problem

We can solve this optimization problem using the method of Lagrange multipliers.
The Lagrangian for the SVM optimization problem can be formulated as follows:

$$ L(w, b, \mu) = \frac{1}{2} \| w \|^2 - \sum_{i=1}^n \mu_i \left( y_i (w^T x_i + b) - 1 \right) $$

where $\mu_i \geq 0$ are the Lagrange multipliers.

To find the optimal solution, we take the partial derivatives of the Lagrangian with respect to $w$, $b$, and $\mu_i$, and set them to zero:

1. **Gradient with respect to \( w \)**:
   $$ \frac{\partial L}{\partial w} = w - \sum_{i=1}^n \mu_i y_i x_i = 0 $$

2. **Gradient with respect to \( b \)**:
   $$ \frac{\partial L}{\partial b} = -\sum_{i=1}^n \mu_i y_i = 0 $$

3. **Complementary slackness condition**:
   $$ \mu_i (y_i (w^T x_i + b) - 1) = 0 $$

The solution can be obtained by solving these equations, which leads to the optimal weights $w$ and bias $b$ that define the separating hyperplane.

From the first equation, we can express $w$ in terms of $\mu_i$:

$$ w = \sum_{i=1}^n \mu_i y_i x_i $$

The second equation gives us a constraint on $\mu_i$:

$$ \sum_{i=1}^n \mu_i y_i = 0 $$

Substituting these back into the Lagrangian and simplifying, we get the dual formulation:

## Dual Formulation

The dual formulation of the SVM problem can be expressed as:

$$ \max_{\mu} \sum_{i=1}^n \mu_i - \frac{1}{2} \sum_{i=1}^n \sum_{j=1}^n \mu_i \mu_j y_i y_j (x_i^T x_j) $$

subject to:

$$ \sum_{i=1}^n \mu_i y_i = 0 $$

$$ \mu_i \geq 0 \quad \text{for all } i $$

## Finding $w$

The dual problem can be solved using quadratic programming techniques. Once we have the optimal $\mu_i$, we can recover $w$ using the equation:

$$ w = \sum_{i=1}^n \mu_i y_i x_i $$

## Finding $b$

To find $b$, we can use any support vector (a point where $\mu_i > 0$) and the fact that for these points, $y_i(w^T x_i + b) = 1$.

## Decision Function

The decision function for classifying new points becomes:

$$ f(x) = \text{sign}\left(\sum_{i=1}^n \mu_i y_i (x_i^T x) + b\right) $$

where only the support vectors (points with $\mu_i > 0$) contribute to the sum.






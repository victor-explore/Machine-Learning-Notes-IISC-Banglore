# Linear regression

Let the data be $(x_i, y_i)$ for $i = 1, 2, ..., n$ where $x_i \in \mathbb{R}^d$ and $y_i \in \mathbb{R}$.

then we can model $y_i$ as:


$$Y = \beta_0 + \beta_1X + \varepsilon$$

Where:
- $Y \in \mathbb{R}$ is the dependent variable
- $X \in \mathbb{R}^d$ is the independent variable (or feature vector in higher dimensions)
- $\beta_0$ is the y-intercept (bias term)
- $\beta_1$ is the slope (or coefficient vector in higher dimensions)
- $\varepsilon$ is the error term

## Ideal regressor

For mean squared error loss, the ideal regressor is defined as:

$$h^*(x) = E[Y|X=x]$$

Where $E[Y|X=x]$ is the conditional expectation of $Y$ given $X=x$.

## Derivation of ideal regressor
To derive the ideal regressor for squared error loss, we need to find the function h(x) that minimizes the expected squared error:

$$h^* = \arg\min_h E[(Y - h(X))^2]$$

Let's expand this expectation:

$$E[(Y - h(X))^2] = E[Y^2] - 2E[Yh(X)] + E[h(X)^2]$$

To minimize this, we differentiate with respect to h(x) and set it to zero:

$$\frac{\partial}{\partial h(x)} E[(Y - h(X))^2] = -2E[Y|X=x] + 2h(x) = 0$$

Solving for $h^*(x)$:

$$h^*(x) = E[Y|X=x]$$

This shows that the ideal regressor for squared error loss is indeed the conditional expectation of $Y$ given $X=x$.

Now, let's derive this further using our linear model:

$$E[Y|X=x] = E[\beta_0 + \beta_1X + \varepsilon|X=x]$$

Using the linearity of expectation:

$$E[\beta_0 + \beta_1X + \varepsilon|X=x] = E[\beta_0|X=x] + E[\beta_1X|X=x] + E[\varepsilon|X=x]$$

Simplifying:

1. $E[\beta_0|X=x] = \beta_0$ (since $\beta_0$ is a constant)
2. $E[\beta_1X|X=x] = \beta_1x$ (since $X$ is fixed at $x$)
3. $E[\varepsilon|X=x] = 0$ (assuming the error term has zero mean and is independent of $X$)

Therefore:

$$h^*(x) = \beta_0 + \beta_1x$$

## Interpretation

The ideal regressor $h^*(x) = \beta_0 + \beta_1x$ minimizes the expected squared error. It represents the best possible prediction of $Y$ given $X=x$ under the squared error loss, assuming the linear model is correct. This function provides the average value of $Y$ for each value of $X$, effectively capturing the underlying linear relationship between the variables while averaging out the random noise (represented by $\varepsilon$).

## Empirical Risk Minimization

In practice, we don't have access to the true distribution of the data, so we can't directly minimize the expected risk. Instead, we use the empirical risk as an approximation:

$$\hat{R}(\beta) = \frac{1}{n} \sum_{i=1}^n (y_i - (\beta_0 + \beta_1x_i))^2$$

where $n$ is the number of samples in our dataset.

Note that in this formulation, we don't explicitly see the error term $\varepsilon$ that was present in our original model $Y = \beta_0 + \beta_1X + \varepsilon$. This is because the empirical risk is calculated using the observed $y_i$ values, which already incorporate the random error.

To find the optimal parameters $\beta^*$, we minimize this empirical risk:

$$\beta^* = \arg\min_\beta \hat{R}(\beta)$$

This optimization problem has a closed-form solution, which can be derived using linear algebra:

$$\beta^* = (X^T X)^{-1} X^T Y$$
Here, $X$ is the design matrix:

$$X = \begin{bmatrix} 
1 & x_{11} & x_{12} & \cdots & x_{1d} \\
1 & x_{21} & x_{22} & \cdots & x_{2d} \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
1 & x_{n1} & x_{n2} & \cdots & x_{nd}
\end{bmatrix}_{n \times (d+1)}$$

This matrix has $n$ rows (one for each data point) and $d+1$ columns. The first column is all 1s (for the intercept term), and the remaining $d$ columns contain the feature values of our data points.

And $Y$ is the vector of target values:

$$Y = \begin{bmatrix} 
y_1 \\
y_2 \\
\vdots \\
y_n
\end{bmatrix}_{n \times 1}$$

This is a column vector with $n$ rows, containing the scalar $y$ values of our data points.

The solution $\beta^*$ is a $(d+1) \times 1$ vector:

$$\beta^* = \begin{bmatrix} 
\beta_0^* \\
\beta_1^* \\
\vdots \\
\beta_d^*
\end{bmatrix}_{(d+1) \times 1}$$

To retrieve the individual $\beta_1^*$ values from $\beta^*$:

1. $\beta_0^*$ is the first element of $\beta^*$, i.e., $\beta^*[0]$
2. $\beta_1^*$ = $\begin{bmatrix} 
\beta_1^* \\
\vdots \\
\beta_d^*
\end{bmatrix}_{(d) \times 1}$


This solution is known as the Ordinary Least Squares (OLS) estimator. 



# Linear Discriminant Analysis(LDA) from bayesian perspective

Let the data be $D = \{(x_i, y_i)\}_{i=1}^N$ where $x_i \in \mathbb{R}^d$ and $y_i \in \{1, 2\}$

Let's assume the parametric form of the conditional density $p(x|y)$ is:

$$p(x|y=1) \sim N(x; \mu_1, \Sigma)$$
$$p(x|y=0) \sim N(x; \mu_2, \Sigma)$$

where $N(x; \mu, \Sigma)$ denotes a multivariate Gaussian distribution with mean $\mu$ and covariance matrix $\Sigma$. Note that we assume the covariance matrix $\Sigma$ is the same for both classes, which is a key assumption in LDA.

In simple words, we are assuming that the data is distributed as Gaussian in each class with different means but shared covariance matrix.

Also, let's assume that the prior probabilities $P(y=1)$ and $P(y=0)$ are same ie $1/2$.

## Derivation
Let's derive the decision boundary for LDA using the Bayes classifier.

1. Bayes Classifier:
   The Bayes classifier is defined as:
   $$h_B(x) = \begin{cases}
       1 & \text{if } P(y=1 | x=x_i) > P(y=0 | x=x_i) \\
       0 & \text{if } P(y=1 | x=x_i) \leq P(y=0 | x=x_i)
   \end{cases}$$

2. Using Bayes' Rule:
   $$P(y=k|x) = \frac{P(x|y=k) \cdot P(y=k)}{P(x)}$$

3. Decision Rule:
   Choose class 1 if:
   $$P(x|y=1) \cdot P(y=1) > P(x|y=0) \cdot P(y=0)$$

4. Taking logarithms (monotonic transformation):
   $$\log(P(x|y=1)) + \log(P(y=1)) > \log(P(x|y=0)) + \log(P(y=0))$$

5. Substituting Gaussian densities:
   $$-\frac{1}{2}(x-\mu_1)^T\Sigma^{-1}(x-\mu_1) - \frac{d}{2}\log(2\pi) - \frac{1}{2}\log|\Sigma| + \log(P(y=1)) >$$
   $$-\frac{1}{2}(x-\mu_2)^T\Sigma^{-1}(x-\mu_2) - \frac{d}{2}\log(2\pi) - \frac{1}{2}\log|\Sigma| + \log(P(y=0))$$

6. Simplifying:
   $$-\frac{1}{2}(x-\mu_1)^T\Sigma^{-1}(x-\mu_1) + \log(P(y=1)) > -\frac{1}{2}(x-\mu_2)^T\Sigma^{-1}(x-\mu_2) + \log(P(y=0))$$

7. Expanding the quadratic terms:
   $$-\frac{1}{2}(x^T\Sigma^{-1}x - 2\mu_1^T\Sigma^{-1}x + \mu_1^T\Sigma^{-1}\mu_1) + \log(P(y=1)) >$$
   $$-\frac{1}{2}(x^T\Sigma^{-1}x - 2\mu_2^T\Sigma^{-1}x + \mu_2^T\Sigma^{-1}\mu_2) + \log(P(y=0))$$

8. Cancelling out common terms:
   $$\mu_1^T\Sigma^{-1}x - \frac{1}{2}\mu_1^T\Sigma^{-1}\mu_1 + \log(P(y=1)) >$$
   $$\mu_2^T\Sigma^{-1}x - \frac{1}{2}\mu_2^T\Sigma^{-1}\mu_2 + \log(P(y=0))$$

9. Rearranging:
   $$(\mu_1 - \mu_2)^T\Sigma^{-1}x > \frac{1}{2}(\mu_1^T\Sigma^{-1}\mu_1 - \mu_2^T\Sigma^{-1}\mu_2) + \log(P(y=0)/P(y=1))$$

10. Final decision boundary:
    $$h_B(x) = \begin{cases}
    1 & \text{if } w^Tx + w_0 > 0 \\
    0 & \text{otherwise}
    \end{cases}$$
    where:
    $$w = \Sigma^{-1}(\mu_1 - \mu_2)$$
    $$w_0 = -\frac{1}{2}(\mu_1^T\Sigma^{-1}\mu_1 - \mu_2^T\Sigma^{-1}\mu_2) - \log(P(y=0)/P(y=1))$$

## Note
- The decision boundary is linear in $x$. This is the reason it is called Linear Discriminant Analysis.
- The decision boundary $w^Tx + w_0 = 0$ is a hyperplane.
- The decision boundary will not be linear if the covariance matrices are not the same for both classes.




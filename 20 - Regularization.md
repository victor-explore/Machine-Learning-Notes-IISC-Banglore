# Regularization

Regularization is a technique to increase model bias to reduce variance by constraining empirical risk minimization.

regularized empirical risk minimization:

$$\text{Reg } ERM = \min_{h \in H} \hat{R}(h_\theta) \quad \text{s.t. } \Omega(h_\theta) < k$$

here $\Omega(h_\theta) < k$ is the regularization function which is a design choice.

This can be solved using the method of Lagrange multipliers.

$h(x) = \arg\min_{h_\theta \in H} \left( \hat{R}(h_\theta) + \lambda \Omega(\theta) \right)$

here $\lambda$ is the regularization parameter which is a design choice.

## Norm based regularization
When $\Omega(\theta)$ is taken to be the $p$-norm, i.e., $\|\theta\|_p$:

- If $p=1$, then we have L1 (lasso) regularization.
- If $p=2$, then we have L2 (ridge) regularization.

## Importance of regularization
It can be shown that:

$$MLE ≈ ERM$$
$$MAP ≈ \text{reg } ERM$$



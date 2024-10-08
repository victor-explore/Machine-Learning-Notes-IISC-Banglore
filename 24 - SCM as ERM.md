# Support measure machine (SCM) as Empirical Risk Minimization (ERM)

Recall that in support measure machine (SCM), we were trying to maximize the margin between the two classes of data points.

$$ \min_{w,b,\xi} \frac{1}{2} \|w\|^2 + C \sum_{i=1}^n \xi_i $$
$$ \text{s.t. } y_i (w^T x_i + b) \geq 1 - \xi_i \text{ for all } i $$
$$ \xi_i \geq 0 \text{ for all } i $$

## SCM as ERM
This is an optimization problem that can be formulated as a empirical risk minimization (ERM) problem.

$$ \min_{w,b} \frac{1}{2} w^T w + C \sum_{i=1}^n \max(0, 1 - y_i (w^T x_i + b)) $$

This is an ERM problem with:
- the loss function $l(y_i, f(x_i)) = \max(0, 1 - y_i f(x_i))$, is known as the hinge loss.
- the regularization term $\frac{1}{2} w^T w$.


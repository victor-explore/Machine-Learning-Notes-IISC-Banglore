# Risk function (R)

Recall that the loss function $L(y_i, \hat{y_i})$ measures the error between the predicted label and the true label for a single data point.

The risk function $R(h)$ is the expected loss over all data points in the dataset. It is defined as:
$$R(h) = \mathbb{E}_{(x,y) \sim P}[L(y, h(x))]$$

where $P$ is the true data distribution.

## Conditional risk
The conditional risk $R(h|x)$ is the expected loss for a given input $x$. It is defined as:
$$R(h(x)|x) = \mathbb{E}_{y \sim P(y|x)}[L(y, h(x))]$$

Then total risk is the expected value of the conditional risk:
$$R(h) = \mathbb{E}_{x \sim P(x)}[R(h(x)|x)]$$

## Empirical risk

We do not know the true data distribution $P$ and we have access to a dataset $D$ sampled from $P$. Hence we approximate the risk function by the empirical risk:
$$R(h) \approx \frac{1}{n} \sum_{i=1}^n L(y_i, h(x_i))$$

where $\{(x_i, y_i)\}_{i=1}^n$ is the dataset.



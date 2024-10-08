# Logistic regression also known as logit regression or binary classification

Let the data be D = {($x_1$, $y_1$), ($x_2$, $y_2$), ..., ($x_n$, $y_n$)}
where $y_i \in \{0, 1\}$ is the binary label of the feature vector $x_i$.

Then the logistic regression model is defined as:
$$P(y_i = 1 | x_i) = \frac{1}{1 + e^{-w^T x_i+b}}$$
$$P(y_i = 0 | x_i) = \frac{1}{1 + e^{w^T x_i+b}}$$

where $w$ is the parameter vector.

Note that
$$P(y_i = 1 | x_i) + P(y_i = 0 | x_i) = 1$$

## Why define the model like this?

because consider:

$$ p(y=1 | x) = \frac{p(x|y=1)}{p(x|y=1) + p(x|y=0)}$$

$$ p(y=1 | x) = \frac{1}{1 + \frac{p(x|y=0)}{p(x|y=1)}}$$

if:
- $y_i \in \{0, 1\}$ and the priors ie $P(y_i = 1)$ and $P(y_i = 0)$ are equal
- And $p(x_i | y_i = 0)$ and $p(x_i | y_i = 1)$ follows an  gaussian distribution with different means and equal covariance matrix
then this becomes:
$$P(y_i = 1 | x_i) = \frac{1}{1 + e^{-w^T x_i+b}}$$
$$P(y_i = 0 | x_i) = \frac{1}{1 + e^{w^T x_i+b}}$$

## How to find the parameters $w$ and $b$?

For the binary classification problem, use logistic regression as hypothesis function and cross-entropy loss function as the loss function and perform gradient descent to find the parameters $w$ and $b$.

## Why not use hard labels (0 or 1)?
Using hard labels (0 or 1) directly in backpropagation can lead to several issues, hence we use soft labels. Soft labels provide a continuous probability distribution over the classes, allowing for smoother gradients and more stable training. This approach enables the model to capture uncertainty and learn more nuanced decision boundaries compared to hard binary classifications.

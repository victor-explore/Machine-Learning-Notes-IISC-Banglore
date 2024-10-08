# Bayes classifier

Let the data be D = {($x_1$, $y_1$), ($x_2$, $y_2$), ..., ($x_n$, $y_n$)}
where $y_i$ is the label of the feature vector $x_i$.

Let the hypothesis space be $H = \{h | h: X \to Y\}$.

The Bayes classifier is defined as:
$$h_B(x) = \begin{cases}
    1 & \text{if } P(y=1 | x=x_i) > P(y=0 | x=x_i) \\
    0 & \text{if } P(y=1 | x=x_i) \leq P(y=0 | x=x_i)
\end{cases}$$

## Notations
- $P(y)$ is also called the prior probability of class $y$ or class prior probability.
- $P(x|y)$ is also called the likelihood of $x$ given class $y$.
- $P(y|x)$ is also called the posterior probability of class $y$ given $x$.
- $P(x)$ is also called the evidence.


## Bayes classifier is the best classifier for 0-1 loss function
To prove that the Bayes classifier is the best classifier for the 0-1 loss function, we need to show that it minimizes the expected risk (error) for any given input $x$.

Let's define the 0-1 loss function:

$$
L(y, h(x)) = \begin{cases}
    0 & \text{if } y = h(x) \\
    1 & \text{if } y \neq h(x)
\end{cases}
$$

The expected risk for a classifier $h$ is:

$$
R(h) = E[L(y, h(x))] = \int L(y, h(x)) P(x, y) \, dx \, dy
$$

For a given $x$, the conditional risk is:

$$
R(h|x) = E[L(y, h(x)) | x] = P(y=0|x)L(0, h(x)) + P(y=1|x)L(1, h(x))
$$

Now, let's consider two cases:

1. If $h(x) = 0$:
   $$
   R(h|x) = P(y=1|x)
   $$

2. If $h(x) = 1$:
   $$
   R(h|x) = P(y=0|x)
   $$

The Bayes classifier chooses the class that minimizes this conditional risk:

$$
h_B(x) = \arg\min_{y\in\{0,1\}} R(h|x)
$$

This means:
- If $P(y=1|x) > P(y=0|x)$, then $h_B(x) = 1$
- If $P(y=1|x) \leq P(y=0|x)$, then $h_B(x) = 0$

This is exactly the definition of the Bayes classifier we started with.

Since the Bayes classifier minimizes the conditional risk for every $x$, it also minimizes the overall expected risk $R(h)$. Therefore, the Bayes classifier is the optimal classifier for the 0-1 loss function.





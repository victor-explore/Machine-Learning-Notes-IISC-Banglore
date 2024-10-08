# Softmax regression also known multiclass regression

Let the data be D = {($x_1$, $y_1$), ($x_2$, $y_2$), ..., ($x_n$, $y_n$)}
where $y_i \in \{1, 2, ..., K\}$ is the multiclass label of the feature vector $x_i$, and $K$ is the number of classes.

Then the softmax regression model is defined as:
$$P(y_i = k | x_i) = \frac{e^{w_k^T x_i + b_k}}{\sum_{j=1}^K e^{w_j^T x_i + b_j}}$$

where $w_k$ is the parameter vector for class $k$, and $b_k$ is the bias term for class $k$.

Note that
$$\sum_{k=1}^K P(y_i = k | x_i) = 1$$

This ensures that the probabilities for all classes sum up to 1, providing a valid probability distribution over the K classes.

The reason for using softmax regression instead of hard labels and the method to find the parameters $w_k$ and $b_k$ is the same as the reason for using logistic regression.

## How to find the parameters $w_k$ and $b_k$?

For the multiclass classification problem, we use softmax regression as the hypothesis function and cross-entropy loss function as the loss function. We then perform gradient descent to find the parameters $w_k$ and $b_k$ for each class $k$.

## Why use softmax instead of hard labels?

Using softmax instead of hard labels (e.g., one-hot encoding) in multiclass classification offers several advantages:

1. Smooth gradients: Softmax provides a continuous, differentiable output, allowing for smoother gradients during backpropagation. This leads to more stable and efficient training.

2. Probability interpretation: Softmax outputs can be interpreted as probabilities, giving a measure of the model's confidence in its predictions for each class.


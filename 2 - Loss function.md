# Loss function (L)
Let the data be D = {($x_1$, $y_1$), ($x_2$, $y_2$), ..., ($x_n$, $y_n$)}
where $y_i$ is the label of the feature vector $x_i$.

Let the hypothesis function be $h: X \to Y$.

Let $\hat{y_i} = h(x_i)$ be the prediction of this hypothesis function for the feature vector $x_i$, whereas the true label is $y_i$.

Then the loss function $L(y_i, \hat{y_i})$ is a function that measures the error between the predicted label and the true label.

### Desirable properties
1. Non-negative: $L(y_i, \hat{y_i}) \geq 0$
2. Zero if and only if the prediction is correct: $L(y_i, \hat{y_i}) = 0$ if and only if $y_i = \hat{y_i}$
3. Continous and differentiable for all $y_i$ and $\hat{y_i}$ for smooth optimization using gradient descent.

## Examples of loss functions

### 0-1 loss function
$$L(y_i, \hat{y_i}) = \begin{cases} 
0 & \text{if } y_i = \hat{y_i} \\
1 & \text{if } y_i \neq \hat{y_i}
\end{cases}$$

### Square loss function
$$L(y_i, \hat{y_i}) = (y_i - \hat{y_i})^2$$

### Cross-entropy loss function
The cross-entropy loss function is specifically used for classification tasks where $y_i$ and $\hat{y_i}$ represent probabilities. It is defined as:
$$L(y_i, \hat{y_i}) = -y_i \log(\hat{y_i}) - (1 - y_i) \log(1 - \hat{y_i})$$


# Hypothesis space (H)
Let the data be D = {($x_1$, $y_1$), ($x_2$, $y_2$), ..., ($x_n$, $y_n$)}
where $y_i$ is the label of the feature vector $x_i$.

A hypothesis function $h$ is a function that maps the feature vector $x$ to the label $y$, ie 
$$h: X \to Y$$

The hypothesis space is the set of all possible hypothesis functions, denoted as 
$$H = \{h | h: X \to Y\}$$

During the learning process, we try to find the best hypothesis function $h$ from the hypothesis space $H$ that minimizes the error between the predicted label and the true label.
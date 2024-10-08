# Boosting

In bagging, we train each model independently on a random subset of the data. In boosting, we train each model sequentially on the same data, with each subsequent model focusing on correcting the errors of combined previous model by increasing weights of misclassified datapoints

Unlike bagging, each model depends on the previous ones and its contribution to the final prediction is weighted differently.

## Mathematical Formulation of Boosting

Let's define the ensemble model $H_T(x)$ as:

$$ H_T(x) = \sum_{t=1}^T \alpha_t h_t(x) $$

where:
- $h_t(x)$ is the t-th weak learner
- $\alpha_t \in [0,1]$ is the weight of the t-th weak learner
- $T$ is the total number of weak learners

The risk (or error) of the ensemble model $H$ is defined as:

$$ R(H) = \frac{1}{n} \sum_{i=1}^n L(H(x_i), y_i) $$

where:
- $L$ is the loss function
- $(x_i, y_i)$ are the input-output pairs in the dataset
- $n$ is the number of samples

Our goal is to minimize $R(H)$ with respect to $H_T$. We do this by gradient descent over functions:

$$ h_T = \arg\min_{h \in H} R(H_{T-1} + \alpha h) $$

Using Taylor expansion, this can be approximated as:

$$ h_T = \arg\min_{h \in H} R(H_{T-1}) + \alpha \langle \nabla R(H_{T-1}), h \rangle $$

where $\langle \cdot, \cdot \rangle$ denotes the inner product.

Because $R(H_{T-1})$ is fixed, we can ignore it:

$$ h_T = \arg\min_{h \in H} \alpha \cdot\langle \nabla R(H_{T-1}), h \rangle $$

Expanding this further:

$$ h_T = \arg\min_{h \in H} \sum_{i=1}^n \frac{\partial R(H_{T-1}(x_i))}{\partial H_{T-1}(x_i)} \cdot h(x_i) $$

We call $r_i = -\frac{\partial R(H_{T-1}(x_i))}{\partial H_{T-1}(x_i)}$ the pseudo-residuals.

Thus, the problem reduces to:

$$ h_T = \arg\min_{h \in H} \sum_{i=1}^n r_i \cdot h(x_i) $$

This formulation shows that each new weak learner $h_T$ is fit to the pseudo-residuals of the previous ensemble model, effectively focusing on the errors of the combined previous models.




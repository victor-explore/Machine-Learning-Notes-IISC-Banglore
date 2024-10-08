## XGBoost - Gradient boosted regression tree
In XGBoost, the weak learners are decision trees.

The algorithm builds these trees sequentially, with each new tree aiming to correct the errors of the combined previous trees.

1. Recall that $r_i = -\frac{\partial R(H_{T-1}(x_i))}{\partial H_{T-1}(x_i)}$ are the pseudo-residuals.

2. XGBoost formulates the optimization problem for finding the next weak learner as:

   $$ h_T = \arg\min_{h \in H} \sum_{i=1}^n (r_i \cdot h(x_i) + \frac{1}{2} h(x_i)^2) $$

3. We can rewrite this by defining $\hat{y}_i = -r_i$:

   $$ h_T = \arg\min_{h \in H} \sum_{i=1}^n (-\hat{y}_i \cdot h(x_i) + \frac{1}{2} h(x_i)^2) $$

   where $\hat{y}_i = \frac{\partial R(H_{T-1}(x_i))}{\partial H_{T-1}(x_i)}$

4. This formulation is equivalent to:

   $$ h_T = \arg\min_{h \in H} \sum_{i=1}^n (h(x_i) - \hat{y}_i)^2 $$

   which is the standard squared error regression problem with new labels $\hat{y}_i$.

5. In practice, $\hat{y}_i$ is approximated as:

   $$ \hat{y}_i = H_{T-1}(x_i) - y_i $$

   where $y_i$ is the true label and $H_{T-1}(x_i)$ is the prediction of the ensemble up to the previous iteration. This approximation is derived from the first-order Taylor expansion of the gradient when using the squared error loss $L(y_i, H(x_i)) = \frac{1}{2}(y_i - H(x_i))^2$.

## Algorithm
The XGBoost algorithm can be summarized in the following steps:

1. Initialize the model with a constant value:
   $$ H_0(x) = \arg\min_\gamma \sum_{i=1}^n L(y_i, \gamma) $$

2. For t = 1 to T (number of trees):
   
   a. Update labels:
      $$ \hat{y}_i = H_{T-1}(x_i) - y_i $$
      for $i = 1, ..., n$

   b. Fit a regression tree to the updated labels, giving terminal regions $R_j^t$, $j = 1, ..., J_t$

   c. For each terminal region $R_j^t$, compute:
      $$ \gamma_{jt} = \arg\min_\gamma \sum_{x_i \in R_{jt}} L(y_i, H_{t-1}(x_i) + \gamma) $$
   This step computes the optimal value $γ_jt$ (the adjustment) for each leaf region $R_j^t$, minimizing the loss function $L$ for observations in that region. This "boost" is added to the current model predictions in the next step.

   d. Update the model:
      $$ H_t(x) = H_{t-1}(x) + \nu \sum_{j=1}^{J_t} \gamma_{jt} I(x \in R_{jt}) $$
      where $\nu$ is the learning rate (0 < $\nu$ ≤ 1) that controls how much of the new tree's contribution is added to the current model, $\gamma_{jt}$ is the prediction adjustment for each terminal region $R_{jt}$ of the tree, and $I(x \in R_{jt})$ is an indicator function that equals 1 if $x$ belongs to region $R_{jt}$ and 0 otherwise.

3. Output the final model:
   $$ H(x) = H_T(x) = \sum_{t=0}^T \nu \sum_{j=1}^{J_t} \gamma_{jt} I(x \in R_{jt}) $$
   The final model is the sum of all the contributions from the individual trees, with each tree's contribution scaled by the learning rate $\nu$.






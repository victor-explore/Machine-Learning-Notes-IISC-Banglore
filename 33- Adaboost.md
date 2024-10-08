# Adaboost

## Recall
Recall from our discussion on boosting we defined the ensemble model $H_T(x)$ as:

$$ H_T(x) = \sum_{t=1}^T \alpha_t h_t(x) $$

the problem of finding the next weak learner reduces to:

$$ h_T = \arg\min_{h \in H} \sum_{i=1}^n r_i \cdot h(x_i) $$

where 
$$r_i = -\frac{\partial R(H_{T-1}(x_i))}{\partial H_{T-1}(x_i)}$$ 

## Problem setting

Let the dataset be $D = \{(x_i, y_i)\}_{i=1}^n$ where $y_i \in \{-1, 1\}$.

The loss function is given by:

$$L(H(x_i), y_i) = \exp(-y_i H(x_i))$$

The risk of the ensemble model $H$ is given by:

$$R(H) = \frac{1}{n} \sum_{i=1}^n \exp(-y_i H(x_i)).$$

We want to find the model $H_T$ that minimizes this risk.

## Solution
We know:
$$r_i = -\frac{\partial R(H_{T-1}(x_i))}{\partial H_{T-1}(x_i)}$$ 
Let's calculate the pseudo-residuals:

$$ r_i = -\frac{\partial R(H_{T-1}(x_i))}{\partial H_{T-1}(x_i)} = \frac{\partial}{\partial H_{T-1}(x_i)} \left(\frac{1}{n} \sum_{j=1}^n \exp(-y_j H_{T-1}(x_j))\right) $$

$$ = \frac{1}{n} \frac{\partial}{\partial H_{T-1}(x_i)} \exp(-y_i H_{T-1}(x_i)) $$

$$ = \frac{1}{n} \exp(-y_i H_{T-1}(x_i)) (-y_i) $$

$$ = y_i \exp(-y_i H_{T-1}(x_i)) $$

Now, let's define weights $w_i$ for each data point:

$$ w_i = \frac{\exp(-y_i H_{T-1}(x_i))}{\sum_{j=1}^n \exp(-y_j H_{T-1}(x_j))} $$

Note that $\sum_{i=1}^n w_i = 1$.

Using these weights, we can rewrite our optimization problem:

$$ h_T = \arg\min_{h \in H} \sum_{i=1}^n r_i \cdot h(x_i) $$

$$ = \arg\min_{h \in H} \sum_{i=1}^n y_i \exp(-y_i H_{T-1}(x_i)) \cdot h(x_i) $$

$$ = \arg\min_{h \in H} - \sum_{i=1}^n w_i y_i h(x_i) $$

This is equivalent to:

$$ h_T = \arg\max_{h \in H} \sum_{i=1}^n w_i y_i h(x_i) $$

The weak learner $h_T$ is chosen to maximize the weighted correlation between its predictions and the true labels.

In practice, we take any $h$ say MLP, weight the losses for each datapoint $i$ by $w_i$ and then optimize for the weights that minimize the weighted loss.

## Finding step size

After finding $h_T$, we need to determine its weight $\alpha_T$ in the ensemble:
To find the step size $\alpha_{t+1}$, we minimize the risk with respect to $\alpha$:

$$ \alpha_{t+1} = \arg\min_{\alpha} R(H_t + \alpha h) $$

$$ \alpha_{t+1} = \arg\min_{\alpha} \sum_{i=1}^n e^{-y_i(H_t(x_i) + \alpha h(x_i))} $$

Differentiating with respect to $\alpha$ and setting to zero, we get:

$$ \alpha_{t+1} = \frac{1}{2} \log\frac{1-\epsilon}{\epsilon} $$

where $\epsilon = \sum_{i:h(x_i)\neq y_i} w_i$ is the weighted error of $h$.

We then update the ensemble:

$$ H_{t+1}(x) = H_t(x) + \alpha_{t+1} h(x) $$

And update the weights for the next iteration:

$$ w_i^{(t+1)} = w_i^{(t)} e^{-\alpha_{t+1} y_i h(x_i)} $$

These weights are then normalized to sum to 1.

## Algorithm
The AdaBoost algorithm can be summarized in the following steps:

1. Initialize the weights for each training example:
   $$ w_i^{(1)} = \frac{1}{n} \text{ for } i = 1, ..., n $$

2. For t = 1 to T (number of weak learners):
   
   a. Train a weak learner $h_t$ using the weighted training data:
      - Choose any suitable model as the weak learner, such as a decision tree, MLP, or logistic regression.
      - For each training example $i$, weight its contribution to the loss function by $w_i^{(t)}$.
      - If using a decision tree:
        - Modify the splitting criterion to account for sample weights.
        - When calculating impurity measures (e.g., Gini index or entropy), use weighted sums.
      - If using an MLP or logistic regression:
        - Modify the loss function to include sample weights. For example, with binary cross-entropy loss:
          $$ L = -\sum_{i=1}^n w_i^{(t)} [y_i \log(h_t(x_i)) + (1-y_i) \log(1-h_t(x_i))] $$
      - Optimize the parameters of $h_t$ to minimize this weighted loss function.
      - The resulting $h_t$ will focus more on correctly classifying examples with higher weights.
   
   b. Calculate the weighted error of $h_t$:
      $$ \epsilon_t = \sum_{i:h_t(x_i)\neq y_i} w_i^{(t)} $$
   
   c. Compute the weight of the weak learner:
      $$ \alpha_t = \frac{1}{2} \log\left(\frac{1-\epsilon_t}{\epsilon_t}\right) $$
   
   d. Update the ensemble:
      $$ H_t(x) = H_{t-1}(x) + \alpha_t h_t(x) $$
   
   e. Update the weights for the next iteration:
      $$ w_i^{(t+1)} = w_i^{(t)} \exp(-\alpha_t y_i h_t(x_i)) $$
      
   f. Normalize the weights:
      $$ w_i^{(t+1)} = \frac{w_i^{(t+1)}}{\sum_{j=1}^n w_j^{(t+1)}} $$

3. Output the final ensemble:
   $$ H(x) = \text{sign}\left(\sum_{t=1}^T \alpha_t h_t(x)\right) $$

This algorithm iteratively builds an ensemble of weak learners, each time focusing more on the examples that were misclassified in previous iterations. The final classifier is a weighted combination of all the weak learners, where the weights are determined by the performance of each weak learner.




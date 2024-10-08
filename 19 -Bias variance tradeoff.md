# Bias variance tradeoff

It can be shown that average risk for square error loss can be decomposed into three components:

$$
R_{\text{avg}}(h) = \mathbb{E}_{P_D} \mathbb{E}_{P_{x,y}} \left[ \left( h_D(x) - \hat{h}(x) \right)^2 \right] \quad \text{// variance (sensitivity to dataset)}
$$

$$
+ \mathbb{E}_{P_{x,y}} \left[ \left( \hat{h}(x) - h^*(x) \right)^2 \right] \quad \text{// bias (how different is avg classifier from optimum classifier)}
$$

$$
+ \mathbb{E}_{P_{x,y}} \left[ \left( h^*(x) - y \right)^2 \right] \quad \text{// irreducible noise (nothing can be done about this)}
$$

where:
- $R_{\text{avg}}(h)$ is the average risk.
- $p_D$ is the distribution of the training data. In simple words,$p_D(D=D_i)$ is the probability of observing the training data $D_i$.
- $\mathbb{E}_{P_D}$ is the expectation over the distribution of the training data.
- $p_{x,y}$ is the distribution of the input-output pairs.
- $\mathbb{E}_{P_{x,y}}$ is the expectation over the input-output pairs.
- $h_D(x)$ is classifier trained on the training data $D$.
- $\hat{h}(x)$ is the average classifier ie $\hat{h}(x) = \mathbb{E}_{P_D}(h_D(x))$
- $h^*(x)$ is the optimal classifier

## Term breakdown

### Variance:
$$
\mathbb{E}_{P_D} \mathbb{E}_{P_{x,y}} \left[ \left( h_D(x) - \hat{h}(x) \right)^2 \right]
$$
Measures how sensitive the learned classifier $h_D(x)$ is to different training datasets $D$. High variance means the model changes significantly with different training sets, making it unstable.

### Bias:
$$
\mathbb{E}_{P_{x,y}} \left[ \left( \hat{h}(x) - h^*(x) \right)^2 \right]
$$
Measures how much the average learned classifier $\hat{h}(x)$ deviates from the optimal classifier $h^*(x)$ that minimizes the error. High bias means the model is systematically inaccurate or underfits.

### Irreducible Noise:
$$
\mathbb{E}_{P_{x,y}} \left[ \left( h^*(x) - y \right)^2 \right]
$$
This term captures the inherent noise in the data $y$. No model can reduce this part, as it reflects randomness or variability in the data that is not related to the features $x$.

## Bias variance tradeoff

1. **Relationship**: 
   - As we decrease bias by making our model more complex (e.g., using more features or a more flexible model), we often increase variance. This means the model may fit the training data very well but perform poorly on unseen data due to overfitting.
   - Conversely, if we increase bias by simplifying the model (e.g., using fewer features or a more rigid model), we may reduce variance, but at the cost of underfitting the training data.

2. **Optimal Point**: 
   - The goal is to find a balance where both bias and variance are minimized, leading to the lowest possible total error. This is often visualized as a U-shaped curve where the total error is minimized at a certain level of model complexity.
  

   ![Bias Variance Tradeoff](Images/Bias%20variance%20tradeoff.JPG)



# Miscellaneous Machine Learning Terms

## Epoch
One complete pass of the entire training dataset for training the model.

## Batch Size
Risk is defined as

$$L = \frac{1}{N} \sum_{i=1}^N l(y_i, \hat{y}_i)$$

Where $l(y_i, \hat{y}_i)$ is a general loss function that measures the discrepancy between the true value $y_i$ and the predicted value $\hat{y}_i$ for each sample.

Instead of calculating the risk over the entire dataset, however calculating the risk over complete dataset is computationally expensive. Hence we calculate the risk over a small subset of the dataset, called a batch to perform back propogation.


## Gradient Descent Variants
### 1. Batch Gradient Descent

- Full dataset: Computes the gradient of the risk function using the entire training dataset.
- Update frequency: Weights are updated after evaluating the entire dataset in one go.
- Efficiency: Can be slow for large datasets as it requires calculating gradients over the whole dataset before updating weights.
- Convergence: More stable gradient leads to smoother convergence.

Mathematically:
$$ \theta = \theta - \eta \nabla_\theta R(\theta) $$
where $\theta$ are the model parameters, $\eta$ is the learning rate, and $R(\theta)$ is the risk function.

### 2. Stochastic Gradient Descent (SGD)

- Single sample: Updates weights using one randomly chosen sample from the dataset at a time.
- Update frequency: Weights are updated after each individual sample, leading to more frequent updates.
- Efficiency: Faster and more efficient for large datasets as each update only requires computing the gradient for one sample.
- Convergence: Can have a noisier path to convergence but may help escape local minima due to randomness.

Mathematically:
$$ \theta = \theta - \eta \nabla_\theta R(\theta; x^{(i)}, y^{(i)}) $$
where $(x^{(i)}, y^{(i)})$ is a single training example.

### 3. Mini-batch Gradient Descent

- Batch of samples: Computes the gradient over a small batch of samples (between full dataset and single sample).
- Update frequency: Weights are updated after evaluating the risk on each mini-batch.
- Efficiency: Faster than batch gradient descent but less noisy than stochastic gradient descent.
- Convergence: Provides a balance between the efficiency of SGD and the stability of batch gradient descent.

Mathematically:
$$ \theta = \theta - \eta \nabla_\theta R(\theta; x^{(i:i+n)}, y^{(i:i+n)}) $$
where $(x^{(i:i+n)}, y^{(i:i+n)})$ represents a mini-batch of n training examples.



## Batch Normalization
Normalizes the input layer by adjusting and scaling the activations.

$$ \hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}} $$

Where:
- $x_i$ is the input
- $\mu_B$ is the mini-batch mean
- $\sigma_B^2$ is the mini-batch variance
- $\epsilon$ is a small constant added for numerical stability

Batch Normalization is typically implemented as a non-learnable layer in neural networks, with fixed parameters during inference. During inference, the layer uses the moving averages of mean and variance computed during training, rather than calculating batch statistics, to normalize the inputs.


## Layer Normalization
Layer Normalization normalizes the inputs across the features for each sample in a batch, rather than across the batch for each feature.

Mathematically, for an input x with H features:

$$ \mu = \frac{1}{H} \sum_{i=1}^H x_i $$
$$ \sigma^2 = \frac{1}{H} \sum_{i=1}^H (x_i - \mu)^2 $$
$$ \hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}} $$
$$ y_i = \gamma \hat{x}_i + \beta $$

Where:
- $\mu$ is the mean of the features for a single sample
- $\sigma^2$ is the variance of the features for a single sample
- $\epsilon$ is a small constant for numerical stability
- $\gamma$ and $\beta$ are learnable parameters for scaling and shifting

Implementation:
1. Compute mean and variance across features for each sample
2. Normalize each feature using the computed statistics
3. Scale and shift the normalized values with learnable parameters

Unlike Batch Normalization, Layer Normalization's behavior is the same during training and inference, as it doesn't depend on batch statistics.

## Dropout
- A regularization technique
- Also implemented as a layer in neural networks
- During training:
  - Each neuron is "switched on" with probability p
  - When doing backpropagation, remember if neuron was on or off and update weights accordingly
  - The value of p is generally same across all neurons of same layer but can be different for different layers
- During inference/classification:
  - All neurons are active
  - Output of each neuron is multiplied by probability p to get the output

## N-Fold Cross Validation
- A resampling technique used to evaluate machine learning models
- Dataset is split into N equal parts
- N-1 parts are used for training and 1 part is used for validation
- This process is repeated N times, with each part being used for validation once
- Finally, the performance of the model is evaluated by averaging the results from all N folds

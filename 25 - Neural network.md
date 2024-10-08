# Neural network
A neural network is a computational model inspired by the structure and function of biological neural networks. Mathematically, it can be defined as a series of function compositions:

$$ f(x) = f_L(f_{L-1}(...f_2(f_1(x)))) $$

where $L$ is the number of layers in the network, and each function $f_i$ represents a layer operation.

For a single layer, the operation can be expressed as:

$$ f_i(x) = \sigma(W_i x + b_i) $$

where:
- $W_i$ is the weight matrix for layer $i$
- $b_i$ is the bias vector for layer $i$
- $\sigma$ is a non-linear activation function

The complete neural network can then be written as:

$$ f(x) = \sigma_L(W_L \sigma_{L-1}(W_{L-1} ... \sigma_2(W_2 \sigma_1(W_1 x + b_1) + b_2) ... + b_{L-1}) + b_L) $$

This formulation allows the network to learn complex, non-linear mappings from inputs to outputs through the composition of simpler functions and the application of non-linear activations.



## Non-linear activation functions σ
- Sigmoid or logistic: 
  $$ \sigma(x) = \frac{1}{1 + e^{-x}} $$
  
- Sign function: 
  $$ \text{sign}(x) $$

- Hyperbolic tangent: 
  $$ \text{tanh}(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} $$

- Rectified Linear Unit (ReLU): 
  $$ \text{ReLU}(x) = \max(0, x) $$

  ## Why do we need non-linear activation functions?
  - Without non-linear activation functions, the neural network would be equivalent to a linear model. Let's derive this:

    Consider a neural network with two layers and no activation function:
    
    1. First layer: $y = W_1x + b_1$
    2. Second layer: $z = W_2y + b_2$
    
    Substituting $y$ into the second layer:
    $$z = W_2(W_1x + b_1) + b_2$$
    $$z  = W_2W_1x + W_2b_1 + b_2$$
    
    This can be simplified to:
    $$z = Wx + b$$
    
    Where:
    $$W = W_2W_1$$
    $$b = W_2b_1 + b_2$$
    
    This is the equation of a linear model. Therefore, without non-linear activation functions, regardless of the number of layers, a neural network will always produce a linear transformation of the input.

  - Non-linear activation functions introduce non-linearity into the network, allowing it to learn and represent complex, non-linear relationships in the data.


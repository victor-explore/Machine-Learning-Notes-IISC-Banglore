# Backpropagation

## Notations
![Backpropagation Diagram](Images/Backpropogation.JPG)

- $L$: number of layers in the network.
- $w_{jk}^l$: weight connecting $k^{th}$ neuron of $(l-1)^{th}$ layer to $j^{th}$ neuron of $l^{th}$ layer
- $b_j^l$: bias of $j^{th}$ neuron in $l^{th}$ layer
- $a_j^l$: output of $j^{th}$ neuron in $l^{th}$ layer
- $z_j^l$: preactivation output of $j^{th}$ neuron in $l^{th}$ layer

Here:
$$z_j^l = \sum_k w_{jk}^l a_k^{l-1} + b_j^l$$

Also: 
$$a_j^l = \sigma(z_j^l)$$
$$a_j^l = \sigma(\sum_k w_{jk}^l a_k^{l-1} + b_j^l)$$

## Derivation
Let's assume we use squared loss function:
$$L = \frac{1}{2} \|a^L - y\|^2$$

We define the risk R as the expected loss over the data distribution:
$$R = \mathbb{E}[L]$$

Risk derivative with respect to the output layer:
   $$\frac{\partial R}{\partial a_j^L} = \frac{\partial}{\partial a_j^L} \mathbb{E}[\frac{1}{2} \|a^L - y\|^2] = \mathbb{E}[(a_j^L - y_j)]$$

Let $\delta_j^L = \frac{\partial R}{\partial z_j^L}$ then:
   $$\delta_j^L = \frac{\partial R}{\partial a_j^L} \cdot \frac{\partial a_j^L}{\partial z_j^L} = \mathbb{E}[(a_j^L - y_j)] \cdot \sigma'(z_j^L)$$

Error term for the output layer (using element-wise product ⊙):
   $$\delta^L = \nabla_a R \odot \sigma'(z^L)$$

Error term for hidden layers:
   $$\frac{\partial R}{\partial z_j^l} = \sum_k \frac{\partial R}{\partial z_k^{l+1}} \cdot \frac{\partial z_k^{l+1}}{\partial a_j^l} = \sum_k \delta_k^{l+1} \cdot w_{jk}^{l+1} \cdot \sigma'(z_j^l)$$
   because there are k neurons in $l+1^{th}$ layer.

This can be written in vector form as:
   $$\delta^l = ((w^{l+1})^T \delta^{l+1}) \odot \sigma'(z^l)$$

Gradient of the risk with respect to weights:
   $$\frac{\partial R}{\partial w_{jk}^l} = \frac{\partial R}{\partial z_j^l} \cdot \frac{\partial z_j^l}{\partial w_{jk}^l} = \delta_j^l \cdot a_k^{l-1}$$

Gradient of the risk with respect to biases:
   $$\frac{\partial R}{\partial b_j^l} = \frac{\partial R}{\partial z_j^l} \cdot \frac{\partial z_j^l}{\partial b_j^l} = \delta_j^l$$


These derivatives form the basis of the backpropagation algorithm, allowing us to compute the gradients needed for updating the weights and biases in the neural network.

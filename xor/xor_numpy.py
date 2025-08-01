# Packages
import numpy as np
from f import *

# Same results across different platforms
np.random.seed(1234)

# Dataset
x = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([0, 1, 1, 0])

# Parameters
n_inputs = 2
n_outputs = 2
n_neurons = 3

# Weights
W1 = np.random.randn(n_inputs, n_neurons)
b1 = np.random.randn(n_neurons)
W2 = np.random.randn(n_neurons, n_outputs)
b2 = np.random.randn(n_outputs)

# Packed weights
params = [W1, b1, W2, b2]

# Train loop
for i in range(1000):
  # Forward pass
  l1 = x @ W1 + b1
  a1 = np.tanh(l1)
  l2 = a1 @ W2 + b2
  
  # Evaluate model
  loss = cross_entropy(l2, y)
  print(loss)
  
  # Backward pass
  l2_grad = cross_entropy_grad(l2, y)
  W2_grad = a1.T @ l2_grad
  b2_grad = np.sum(l2_grad, axis=0, keepdims=True)[0]
  a1_grad = l2_grad @ W2.T
  l1_grad = a1_grad * tanh_grad(l1)
  W1_grad = x.T @ l1_grad
  b1_grad = np.sum(l1_grad, axis=0, keepdims=True)[0]
  
  # Pack grads
  grads = [W1_grad, b1_grad, W2_grad, b2_grad]
  
  # Update weights
  for param, grad in zip(params, grads):
    param += -0.1 * grad

# Print number of params
num_neurons = n_neurons + n_outputs
num_params = sum(p.size for p in params)
print(f'\nNet has {num_neurons} neurons and {num_params} trainable params\n')

# Test loop
for i in x:
  print(f'Input: {int(i[0].item())} {int(i[1].item())} ', end='')

  # Forward pass
  l1 = i @ W1 + b1
  a1 = np.tanh(l1)
  l2 = a1 @ W2 + b2
  
  # Convert logits to probabilities
  probs = softmax(l2)
  
  # Extract index prediction index
  result = np.argmax(probs).item()
  print(f'prediction is {result}')

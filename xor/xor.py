# Packages
import torch
import torch.nn.functional as F

# Always generate same results
torch.manual_seed(1234)

# Input data
x = torch.tensor([
  [0, 0],
  [0, 1],
  [1, 0],
  [1, 1]
], dtype=torch.float)

# Output data or labels
y = torch.tensor([0, 1, 1, 0])

# Number of input (columns) in training data
n_inputs = x.shape[1]

# Number of output neurons (possible outcomes)
n_outputs = 2

# Number of neurons in a hidden layer
n_neurons = 3

# Single layer perceptron (CORRECTION!!!)
#W = torch.randn((n_inputs, n_outputs), requires_grad=True)
#b = torch.randn((n_outputs,), requires_grad=True)

# Init weights aka trainable parameters (Multi-layer perceptron)
W1 = torch.randn((x.shape[1], n_neurons), requires_grad=True)
b1 = torch.randn((n_neurons), requires_grad=True)
W2 = torch.randn((n_neurons, n_outputs), requires_grad=True)
b2 = torch.randn((n_outputs), requires_grad=True)

# Weights in a list for bulk update
params = [W1, b1, W2, b2]

# Gradient descent aka "training" the net
for i in range(1000):
  # Forward pass
  input_layer = x @ W1 + b1           # compute input layer
  h = torch.tanh(input_layer)         # activation function to add non-linearity
  logits = h @ W2 + b2                # compute output layer "scores"
  loss = F.cross_entropy(logits, y)   # evaluate model
  print(f'{loss.item()}')
  
  # Backward pass
  for param in params:                # for all parameters
    param.grad = None                 # reset gradients
  loss.backward()                     # backpropagate gradients
  
  # Update
  for param in params:                # for all parameters
    param.data += -0.1 * param.grad   # update weights depending on gradients

# Print number of params
num_neurons = n_neurons + n_outputs
num_params = sum(p.nelement() for p in params)
print(f'\nNet has {num_neurons} neurons and {num_params} trainable params\n')

# Inference (testing model predictions)
with torch.no_grad():                   # no gradients needed because we are not training
  for i in x:                           # for all inputs
    print(f'Input: {int(i[0].item())} {int(i[1].item())} ', end='')
  
    # Forward pass
    input_layer = i @ W1 + b1
    h = torch.tanh(input_layer)
    logits = h @ W2 + b2
    
    # Convert logits to probabilities
    probs = logits.softmax(0)
    
    # Extract index prediction index
    result = torch.argmax(probs).item()
    print(f'prediction is {result}')

import torch
import torch.nn.functional as F

torch.manual_seed(1234)

x = torch.tensor([[0,0],[0,1],[1,0],[1,1]], dtype=torch.float)
y = torch.tensor([0, 1, 1, 0])

W1 = torch.randn((2,3), requires_grad=True)
b1 = torch.randn((3), requires_grad=True)
W2 = torch.randn((3,2), requires_grad=True)
b2 = torch.randn((2), requires_grad=True)

params = [W1, b1, W2, b2]

l1 = x @ W1 + b1
a1 = torch.tanh(l1)
l2 = a1 @ W2 + b2

loss = F.cross_entropy(l2, y)
loss.backward()

# Final NN expression               -> F.cross_entropy(l2, y)
# STEP 1: strip F.cross_entropy     -> l2
#         get l2 gradient
# STEP 3: unfold l2                 -> a1 @ W2 + b2
#         get W2, b2 gradients
# STEP 4: strip W2, b2              -> a1
#         unfold a1                 -> torch.tanh(l1)
#         get a1, l1 gradients
# STEP 5: strip torch.tanh          -> l1
#         unfold l1                 -> x @ W1 + b1
#         get W1, b1 gradients

def cross_entropy_grad(m, batch_size):      
  m_grad = torch.softmax(m, dim=1)
  m_grad[range(batch_size), y] -= 1
  m_grad /= batch_size
  return m_grad

def tanh_grad(m):
  m_grad = (1 - torch.tanh(m) ** 2)
  return m_grad

l2_grad = cross_entropy_grad(l2, 4)

W2_grad = a1.T @ l2_grad
b2_grad = l2_grad.sum(0, keepdim=True)

a1_grad = l2_grad @ W2.T
l1_grad = a1_grad * tanh_grad(l1)

W1_grad = x.T @ l1_grad
b1_grad = l1_grad.sum(0, keepdim=True)

print('W2.grad:', W2.grad, 'W2_grad:', W2_grad, sep='\n')
print('b2.grad:', b2.grad, 'b2_grad:', b2_grad, sep='\n')

print('W1.grad:', W1.grad, 'W1_grad:', W1_grad, sep='\n')
print('b1.grad:', b1.grad, 'b1_grad:', b1_grad, sep='\n')

import numpy as np

def softmax(logits, axis=-1):
  exps = np.exp(logits - np.max(logits, axis=axis, keepdims=True))
  return exps / np.sum(exps, axis=axis, keepdims=True)

def cross_entropy(logits, y):
  logits = logits - np.max(logits, axis=1, keepdims=True)
  counts = np.exp(logits)
  probs = counts / np.sum(counts, axis=1, keepdims=True)
  correct_class_probs = probs[np.arange(len(y)), y]
  loss = -np.mean(np.log(correct_class_probs + 1e-12))
  return loss

def cross_entropy_grad(logits, y):
  batch_size = y.shape[0]
  logits = logits - np.max(logits, axis=1, keepdims=True)  # numerical stability
  exp_logits = np.exp(logits)
  probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
  grad = probs.copy()
  grad[np.arange(batch_size), y] -= 1
  grad /= batch_size
  return grad

def tanh_grad(m):
  m_grad = (1 - np.tanh(m) ** 2)
  return m_grad

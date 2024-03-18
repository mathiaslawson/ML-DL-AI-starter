#  A typical PyTorch pipeline looks like this:

# Design model (input, output, forward pass with different layers)
# Construct loss and optimizer
# Training loop:
# Forward = compute prediction and loss
# Backward = compute gradients
# Update weights


import torch
import torch.nn as nn

# Linear regression
# f = w * x 
# here : f = 2 * x

# 0) Training samples, watch the shape!
X = torch.tensor([[1], [2], [3], [4], [5], [6], [7], [8]], dtype=torch.float32)
Y = torch.tensor([[2], [4], [6], [8], [10], [12], [14], [16]], dtype=torch.float32)

n_samples, n_features = X.shape
print(f'n_samples = {n_samples}, n_features = {n_features}')

# 0) create a test sample
X_test = torch.tensor([5], dtype=torch.float32)
import torch
from torch.nn import functional as F

a = torch.arange(10)
print(a)

y = F.one_hot(a, num_classes=10)
z = torch.randn(10, 100)
combined = torch.cat([z, y], dim=1)
print(combined)
print(combined.shape)
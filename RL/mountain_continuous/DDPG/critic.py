import torch
from torch import nn

class Critic(nn.Module):
    def __init__(self, dim_observe, dim_action) -> None:
        super().__init__()

        self.fc1 = nn.Linear(dim_observe+dim_action, 16)
        self.fc2 = nn.Linear(16, 32)
        self.fc3 = nn.Linear(32, 1)
        
        self.activation = nn.GELU()

    def forward(self, state, action):
        x = torch.cat((state, action), dim=1)
        x = self.fc1(x)
        x = self.activation(x)

        x = self.fc2(x)
        x = self.activation(x)

        x = self.fc3(x)
        return x
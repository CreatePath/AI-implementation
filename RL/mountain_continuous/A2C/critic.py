from torch import nn

class Critic(nn.Module):
    def __init__(self, dim_observe) -> None:
        super().__init__()

        self.fc1 = nn.Linear(dim_observe, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 1)
        
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)

        x = self.fc2(x)
        x = self.activation(x)

        x = self.fc3(x)
        return x
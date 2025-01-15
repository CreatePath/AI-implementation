from torch import nn

class Actor(nn.Module):
    def __init__(self, dim_observe, dim_action) -> None:
        super().__init__()

        self.fc1 = nn.Linear(dim_observe, 16)
        self.fc2 = nn.Linear(16, 32)
        self.fc3 = nn.Linear(32, dim_action)

        self.activation = nn.GELU()
        self.final_layer = nn.Tanh()

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)

        x = self.fc2(x)
        x = self.activation(x)

        x = self.fc3(x)
        x = self.final_layer(x)
        return x
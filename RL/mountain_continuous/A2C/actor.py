from torch import nn

class Actor(nn.Module):
    def __init__(self, dim_observe, dim_mu, dim_sigma) -> None:
        super().__init__()

        self.fc1 = nn.Linear(dim_observe, 32)
        self.fc2 = nn.Linear(32, 32)
        self.mu = nn.Linear(32, dim_mu)
        self.sigma = nn.Linear(32, dim_sigma)

        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)

        x = self.fc2(x)
        x = self.activation(x)

        mu = self.mu(x)
        sigma = self.sigma(x)

        return mu, sigma
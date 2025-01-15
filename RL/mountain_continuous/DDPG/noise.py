import torch

class OU_Noise:
    def __init__(self, action_size: int, mu: float, theta: float, sigma: float, device):
        self.action_size = action_size
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.device = device
        self.reset()

    def reset(self):
        self.X = torch.ones(self.action_size, device=self.device) * self.mu

    def sample(self):
        dx = self.theta * (self.mu - self.X)
        dx += self.sigma * torch.randn(len(self.X), device=self.device)
        self.X += dx
        return self.X
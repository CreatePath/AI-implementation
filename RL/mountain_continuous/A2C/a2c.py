import torch
from torch import nn
from torch.distributions import Distribution
from torch import optim
from torch.nn import functional as F

class A2C:
    def __init__(self,
                 gamma: float,
                 pi: nn.Module,
                 v: nn.Module,
                 opt_pi: optim.Optimizer,
                 opt_v: optim.Optimizer,
                 pdf: Distribution) -> None:
        self.gamma = gamma
        self.pi = pi
        self.v = v
        self.optimizer_pi = opt_pi
        self.optimizer_v = opt_v
        self.pdf = pdf
        self.data = []
        self.train_history = {"V": [], "PI": []}

    def get_action(self, x: torch.Tensor):
        mu, sigma = self.pi(x)
        dist = self.pdf(mu, sigma)
        action = dist.sample()
        action = torch.clip(action, -1.0, 1.0)
        log_prob = dist.log_prob(action)
        return action, log_prob

    def update(self, state, log_prob, reward, next_state, done_mask):
        target = reward + self.gamma * self.v(next_state) * done_mask
        target.detach()

        v = self.v(state)
        loss_v = F.smooth_l1_loss(v, target)

        delta = target - v
        delta = delta.detach()

        loss_pi = -log_prob * delta

        self.optimizer_v.zero_grad()
        self.optimizer_pi.zero_grad()
        loss_v.backward()
        loss_pi.backward()
        self.optimizer_v.step()
        self.optimizer_pi.step()

        self.train_history["V"].append(loss_v.item())
        self.train_history["PI"].append(loss_pi.item())
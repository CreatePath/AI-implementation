import torch
from torch import nn
from torch.optim import Optimizer

from noise import OU_Noise
from memory import Memory

def copy_net(tgt: nn.Module, src: nn.Module):
    for t_param, s_param in zip(tgt.parameters(), src.parameters()):
        t_param.data.copy_(s_param.data.clone())

class DDPG:
    def __init__(self,
                 gamma: float,
                 tau: dict[str, float],
                 networks: dict[str, nn.Module],
                 loss_critic: nn.Module,
                 optimizers: dict[str, Optimizer],
                 noise_generator: OU_Noise,
                 is_train: bool = True,
                 eps: float = 1.0,
                 eps_decay: float = 0.02) -> None:

        self.gamma = gamma
        self.tau_actor = tau["actor"]
        self.tau_critic = tau["critic"]
        self.is_train = is_train
        self.eps = eps
        self.eps_decay = eps_decay

        self.actor = networks["actor"]
        self.critic = networks["critic"]
        self.target_actor = networks["target_actor"]
        self.target_critic = networks["target_critic"]

        copy_net(self.target_actor, self.actor)
        copy_net(self.target_critic, self.critic)

        self.loss_critic = loss_critic

        self.opt_a = optimizers["actor"]
        self.opt_c = optimizers["critic"]

        self.noise_generator = noise_generator

        self.loss_history = {"loss_critic": [],
                             "loss_actor": [],}
    
    def get_action(self, state: torch.Tensor):
        if self.is_train:
            action = self.explore(state)
        else:
            with torch.no_grad():
                action = self.target_actor(state)
        return torch.clip(action, -1.0, 1.0)
    
    def explore(self, state: torch.Tensor):
        action = self.actor(state)
        noise = self.noise_generator.sample()
        action += max(self.eps, 0) * noise
        return action
    
    def train(self, memory: Memory):
        if len(memory) < memory.batch_size:
            return
        
        states, actions, rewards, next_states, dones = memory.sample()

        self._update_critic(states, actions, rewards, next_states, dones)
        self._update_actor(states)
        self._update_target(self.target_actor, self.actor, self.tau_actor)
        self._update_target(self.target_critic, self.critic, self.tau_critic)
    
    def _update_critic(self, states, actions, rewards, next_states, dones):
        target_q = self._calculate_target_q(next_states, rewards, dones)
        curr_q = self.critic(states, actions)

        loss_critic = self.loss_critic(curr_q, target_q)

        self.opt_c.zero_grad()
        loss_critic.backward()
        self.opt_c.step()

        self.loss_history["loss_critic"].append(loss_critic.item())
    
    def _calculate_target_q(self, next_states, rewards, dones):
        with torch.no_grad():
            next_actions = self.target_actor(next_states)
            target_q = self.target_critic(next_states, next_actions)
            target_q = rewards + (1 - dones) * self.gamma * target_q
        return target_q
    
    def _update_actor(self, states):
        actions = self.actor(states)
        loss_actor = -self.critic(states, actions).mean()
        self.opt_a.zero_grad()
        loss_actor.backward()
        self.opt_a.step()
        self.loss_history["loss_actor"].append(loss_actor.item())
    
    def _update_target(self, target: nn.Module, src: nn.Module, tau: float):
        for t_param, s_param in zip(target.parameters(), src.parameters()):
            t_param.data.copy_(tau * s_param.data + (1.0 - tau) * t_param.data)
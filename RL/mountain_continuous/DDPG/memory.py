import torch
import numpy as np
import random
from collections import deque

class Memory:
    def __init__(self, capacity: int, batch_size, device="cpu") -> None:
        self.memory = deque([], maxlen=capacity)
        self.batch_size = batch_size
        self.device = device
    
    def insert(self, data):
        self.memory.append(data)
    
    def sample(self):
        transitions = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*transitions)

        states = np.array(states)
        rewards = np.array(rewards)
        next_states = np.array(next_states)        
        dones = np.array(dones)

        return (torch.tensor(states, dtype=torch.float32, device=self.device),
                torch.tensor(actions, dtype=torch.float32, device=self.device).unsqueeze(1),
                torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1),
                torch.tensor(next_states, dtype=torch.float32, device=self.device),
                torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1),)

    def reset(self):
        self.memory.clear()
    
    def __len__(self):
        return len(self.memory)
import matplotlib.pyplot as plt
import gymnasium
from gymnasium import Env

import torch
from torch import optim
from torch.nn import MSELoss

import random
import numpy as np

from ddpg import DDPG
from actor import Actor
from critic import Critic

from memory import Memory
from noise import OU_Noise

SEED = 42
GOAL_STATE_REWARD = 100
MEMORY_CAP = 1_000_000
PATH = "./model/ddpg.pkl"

def train(agent: DDPG, env: Env, episodes: int, batch_size: int, device: str):
    # TODO: if there is a ddpg.pkl file, then load data
    train_history = []
    test_history = []
    memory = Memory(capacity=MEMORY_CAP, batch_size=batch_size, device=device)
    for i in range(episodes):
        train_reward = train_one_episode(agent, env, memory, device)
        train_history.append(train_reward)

        test_reward = test_one_episode(agent, env, device)
        test_history.append(test_reward)

        print("Episode {} - Memory size: {}, Train score: {}, Test score: {}".format(i+1, len(memory), train_reward, test_reward))
        if agent.loss_history["loss_actor"]:
            print("Actor Loss: {}, Critic Loss: {}".format(agent.loss_history["loss_actor"][-1],
                                                           agent.loss_history["loss_critic"][-1]))

        
        if i % 100 == 0:
            save_checkpoint(agent, i, train_history, test_history, memory)
    
    return train_history

def train_one_episode(agent: DDPG, env: Env, memory: Memory, device: str):
    total_reward = 0
    agent.is_train = True
    state, _ = env.reset(seed=SEED)
    while True:
        action, reward, next_state, done, truncated = simulate(agent, env, state, device)
        total_reward += reward

        memory.insert((state, action, reward, next_state, done))
        agent.train(memory)

        state = next_state

        if done or truncated:
            agent.eps -= agent.eps_decay
            break
    
    return total_reward

def test_one_episode(agent: DDPG, env: Env, device: str):
    total_reward = 0
    agent.is_train = False
    state, _ = env.reset(seed=SEED)
    while True:
        _, reward, next_state, done, truncated = simulate(agent, env, state, device)
        total_reward += reward
        state = next_state

        if done or truncated:
            break
    
    return total_reward

def simulate(agent: DDPG, env: Env, state: np.ndarray, device: str):
    action = agent.get_action(torch.tensor(state, device=device))
    next_state, reward, terminated, truncated, _ = env.step(action.detach().cpu().numpy())
    done = 1.0 if terminated else 0.0
    reward = get_reward(next_state, done)
    return action, reward, next_state, done, truncated

def get_reward(next_state, done):
    if done:
        return GOAL_STATE_REWARD
    pos, velocity = next_state
    return pos / 10 + velocity

def save_checkpoint(agent: DDPG, episode: int, train_history: list[float], test_history: list[float], memory: list[tuple]):
    torch.save({"actor_state_dict": agent.actor.state_dict(),
                "critic_state_dict": agent.critic.state_dict(),
                "target_actor_state_dict": agent.target_actor.state_dict(),
                "target_critic_state_dict": agent.target_critic.state_dict(),
                "episode": episode,
                "train_history": train_history,
                "test_history": test_history,
                "memory": memory,
                "loss_critic": agent.loss_history["loss_critic"], 
                "loss_actor": agent.loss_history["loss_actor"],}, PATH)

def main():
    env = gymnasium.make("MountainCarContinuous-v0")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    # Hyper Parameters
    episodes = 10_000
    dim_observe = env.observation_space.shape[0]
    dim_action = env.action_space.shape[0]
    gamma = 0.99
    tau_actor = 0.4
    tau_critic = 0.3
    mu = 0
    theta = 0.1
    sigma = 0.5
    lr_actor = 1e-4
    lr_critic = 1e-3
    batch_size = 128
    noise_eps_decay = 0.005

    actor = Actor(dim_observe, dim_action).to(device)
    critic = Critic(dim_observe, dim_action).to(device)
    target_actor = Actor(dim_observe, dim_action).to(device)
    target_critic = Critic(dim_observe, dim_action).to(device)

    opt_a = optim.Adam(actor.parameters(), lr=lr_actor)
    opt_c = optim.Adam(critic.parameters(), lr=lr_critic)

    tau = {"actor": tau_actor,
           "critic": tau_critic,}

    networks = {"actor": actor,
                "critic": critic,
                "target_actor": target_actor,
                "target_critic": target_critic}

    optimizers = {"actor": opt_a,
                  "critic": opt_c}

    loss_critic = MSELoss()
    noise_generator = OU_Noise(dim_action, mu, theta, sigma, device)
    agent = DDPG(gamma, tau, networks, loss_critic, optimizers, noise_generator, eps_decay=noise_eps_decay)

    results = train(agent, env, episodes, batch_size, device)

    xaxis = [i for i in range(len(results))]
    plt.title("Train History")
    plt.plot(xaxis, results)
    plt.savefig("./results/train_history.png")

    xaxis = [i for i in range(len(agent.loss_history["loss_critic"]))]
    plt.title("Critic Loss")
    plt.plot(xaxis, agent.loss_history["loss_critic"])
    plt.savefig('./results/loss_critic.png')

    plt.title("Actor Loss")
    plt.plot(xaxis, agent.loss_history["loss_actor"])
    plt.savefig("./results/loss_actor.png")

if __name__ == "__main__":
    main()
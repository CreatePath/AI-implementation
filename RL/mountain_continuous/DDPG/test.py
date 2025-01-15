import random
import numpy as np
import gymnasium

import matplotlib.pyplot as plt

import torch
from torch.nn import MSELoss
from torch import optim

from noise import OU_Noise

from ddpg import DDPG
from actor import Actor
from critic import Critic

SEED = 42
EPISODES = 1

def evaluate(env: gymnasium.Env, agent: DDPG, device: str, render: bool = False):
    results = []
    agent.is_train = False
    for i in range(EPISODES):
        state, _ = env.reset(seed=SEED)
        total_reward = 0
        while True:
            if render:
                env.render()
            action = agent.get_action(torch.tensor(state, device=device))
            next_state, reward, terminated, truncated, _ = env.step(action.detach().cpu().numpy())
            done = 1.0 if terminated else 0.0
            reward /= 100
            total_reward += reward

            state = next_state

            if terminated or truncated:
                print("Episode {} / {} Test score: {}".format(i+1, EPISODES, total_reward))
                results.append(total_reward)
                break
                
    return results


def main():
    env = gymnasium.make("MountainCarContinuous-v0", render_mode="human")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    dim_observe = env.observation_space.shape[0]
    dim_action = env.action_space.shape[0]
    gamma = 0.99
    tau_actor = 0.003
    tau_critic = 0.001
    mu = 5e-4
    theta = 0.1
    sigma = 0.1
    lr_actor = 1e-4
    lr_critic = 1e-4

    checkpoint = torch.load("./model/ddpg.pkl")
    print("checkponit is loaded")

    actor = Actor(dim_observe, dim_action).to(device)
    actor.load_state_dict(checkpoint["actor_state_dict"])

    target_actor = Actor(dim_observe, dim_action).to(device)
    target_actor.load_state_dict(checkpoint["target_actor_state_dict"])

    critic = Critic(dim_observe, dim_action).to(device)
    critic.load_state_dict(checkpoint["critic_state_dict"])

    target_critic = Critic(dim_observe, dim_action).to(device)
    target_critic.load_state_dict(checkpoint["target_critic_state_dict"])

    print("All networks are loaded")

    opt_a = optim.Adam(actor.parameters(), lr=lr_actor)
    opt_c = optim.Adam(critic.parameters(), lr=lr_critic)

    tau = {"actor": tau_actor,
           "critic": tau_critic}

    networks = {"actor": actor,
                "target_actor": target_actor,
                "critic": critic,
                "target_critic": target_critic}
    
    optimizers = {"actor": opt_a,
                  "critic": opt_c}

    loss_critic = MSELoss()
    noise_generator = OU_Noise(dim_action, mu, theta, sigma, device)
    agent = DDPG(gamma, tau, networks, loss_critic, optimizers, noise_generator)
    render = True

    print("agent is loaded")
    print("render:", render)

    print("Start Test")
    results = evaluate(env, agent, device, render)

    train_history = checkpoint["train_history"]
    test_history = checkpoint["test_history"]
    actor_loss = checkpoint["loss_actor"]
    critic_loss = checkpoint["loss_critic"]

    plt.title("Rewards of Target Policy with {} Episodes".format(EPISODES))
    plt.scatter(range(len(results)), results)
    plt.show()

    print("Mean Results:", sum(results) / len(results))

    plt.title("Rewards in Training & Test")
    plt.plot(range(len(train_history)), train_history, label="Train")
    plt.plot(range(len(test_history)), test_history, label="Test")
    plt.legend()
    plt.savefig("./results/history.png")
    plt.show()

    plt.title("Actor Loss in Training")
    plt.plot(range(len(actor_loss)), actor_loss)
    plt.savefig("./results/loss_actor")
    plt.show()
    
    plt.title("Critic Loss in Training")
    plt.plot(range(len(critic_loss)), critic_loss)
    plt.savefig("./results/loss_critic")
    plt.show()

if __name__ == "__main__":
    main()
import matplotlib.pyplot as plt
import gymnasium
from gymnasium import Env

import torch
from torch import optim
from torch.distributions import Normal

from a2c import A2C
from actor import Actor
from critic import Critic

SEED = 42

def train(agent: A2C, env: Env, episodes: int, device: str, render: bool = False):
    results = []
    for i in range(episodes):
        duration_t = 0
        state, _ = env.reset(seed=SEED)
        state = torch.tensor(state, device=device).reshape(1, -1)
        while True:
            if render:
                env.render()
            action, log_prob = agent.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action.cpu().numpy())
            next_state = torch.tensor(next_state, device=device).reshape(1, -1)
            done = 0 if terminated else 1
            reward /= 100

            agent.update(state, log_prob, reward, next_state, done)

            state = next_state
            duration_t += 1

            if terminated or truncated:
                print("Episode {} is finshed in {}".format(i, duration_t))
                results.append(duration_t)
                break
    
    return results

def main():
    env = gymnasium.make("MountainCarContinuous-v0", render_mode="human")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Hyper Parameters
    episodes = 100_000
    dim_observe = env.observation_space.shape[0]
    dim_mu = dim_sigma = env.action_space.shape[0]
    gamma = 0.99
    actor = Actor(dim_observe, dim_mu, dim_sigma).to(device)
    critic = Critic(dim_observe).to(device)
    opt_pi = optim.Adam(actor.parameters(), lr=0.0001)
    opt_v = optim.Adam(critic.parameters(), lr=0.0001)
    pdf = Normal

    agent = A2C(gamma, actor, critic, opt_pi, opt_v, pdf)
    render = True

    results = train(agent, env, episodes, device, render)

    xaxis = [i for i in range(len(results))]
    plt.title("Rewards in Training")
    plt.plot(xaxis, results)
    plt.savefig("./results/rewards.png")

    xaxis = [i for i in range(len(agent.train_history["V"]))]
    plt.title("Critic Loss")
    plt.plot(xaxis, agent.train_history["V"])
    plt.savefig("./results/loss_v.png")

    plt.title("Actor Loss")
    plt.plot(xaxis, agent.train_history["PI"])
    plt.savefig("./results/loss_pi.png")

if __name__ == "__main__":
    main()
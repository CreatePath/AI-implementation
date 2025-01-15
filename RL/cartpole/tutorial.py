import gymnasium as gym
env = gym.make("CartPole-v1", render_mode="human")
observation, info = env.reset(seed=42)

print(env.observation_space.shape)
print(env.action_space.n)

for _ in range(1000):
    observation = env.reset()
    for t in range(100):
        env.render()
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        print(reward)
        if terminated or truncated:
            print("Episode finished after {} timesteps".format(t))
            break
env.close()
import gymnasium as gym
import time

SEED = 42
EPISODES = 1000

env = gym.make("MountainCarContinuous-v0")

for i in range(EPISODES):
    env.reset(seed=SEED)
    for t in range(2000):
        action = env.action_space.sample()
        observations, reward, terminated, truncated, info = env.step(action)
        print(observations)
        print(reward)

        if terminated:
            print("Success in {}".format(t))
            time.sleep(1)
            break
        elif truncated:
            print("truncated in {}".format(t))
            time.sleep(1)
            break

        



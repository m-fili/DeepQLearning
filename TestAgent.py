import numpy as np
import torch
from agent import Agent
from unityagents import UnityEnvironment
import matplotlib.pyplot as plt


env = UnityEnvironment(file_name="Banana_Windows_x86_64/Banana_Windows_x86_64/Banana.exe",
                       no_graphics=True)

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# load the weights from file
agent = Agent(env, gamma=0.99, buffer_size=100000, batch_size=64, learning_rate=5e-4, train_mode=False)
agent.qnetwork.load_state_dict(torch.load('checkpoint.pth'))

env_info = env.reset(train_mode=True)[brain_name]
state = env_info.vector_observations[0]
eps = 0.05
scores = []

for episode in range(100):
    score = 0
    while True:
        action = agent.take_action(state, eps)
        env_info = env.step(action)[brain_name]
        next_state = env_info.vector_observations[0]
        reward = env_info.rewards[0]
        done = env_info.local_done[0]
        score += reward
        state = next_state

        if done:
            break

    scores.append(score)

print("Average Score in 100 consecutive episodes: {}".format(np.mean(scores)))


plt.plot(range(1, 101), scores)
plt.title('Scores in 100 Episodes')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.show()

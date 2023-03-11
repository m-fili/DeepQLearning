import numpy as np
from agent import Agent
from unityagents import UnityEnvironment
import matplotlib.pyplot as plt


env = UnityEnvironment(file_name="Banana_Windows_x86_64/Banana_Windows_x86_64/Banana.exe",
                       no_graphics=True)

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# load the weights from file
agent = Agent(env, gamma=0.99, buffer_size=100000, batch_size=64, learning_rate=5e-4, train_mode=True)
agent.load_weights('checkpoint.pth')

eps = 0.05
scores = []

for episode in range(100):

    score = 0
    env_info = env.reset(train_mode=True)[brain_name]
    state = env_info.vector_observations[0]

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
    print(f'Episode {episode:2}/100 is done (avg. score={np.mean(scores):.2f})', end='\r')


env.close()

print("Average Score in 100 consecutive episodes: {}".format(np.mean(scores)))


plt.plot(range(1, 101), scores, color='blue', alpha=0.85)
plt.hlines(y=np.mean(scores), xmin=1, xmax=100, linestyles='--', color='red', alpha=0.8)
plt.title('Scores in 100 Episodes')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.show()

import numpy as np
from collections import deque
import torch


def dqn(env, agent, n_episodes=100, min_epsilon=0.05, epsilon_decay=0.999,
        t_max=10, update_target_frequency=10,
        update_main_frequency=2):

    TARGET_SCORE = 200.0
    epsilon = 1.0

    # counts the total steps (we use it to see when to train the main qnetwork and when to copy main to target.)
    iteration = 0

    # To store scores
    scores = []
    scores_window = deque(maxlen=100)

    for episode in range(1, n_episodes + 1):

        score = 0
        brain_name = env.brain_names[0]
        env_info = env.reset(train_mode=agent.train_mode)[brain_name]  # reset the environment
        s1 = env_info.vector_observations[0]

        # update epsilon value
        epsilon = max(min_epsilon, epsilon * epsilon_decay)

        for t in range(t_max):

            # take action when in state S
            action = agent.take_action(s1, epsilon)

            # take next step
            env_info = env.step(action)[brain_name]
            s2 = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]

            # update score in current episode
            score += reward

            # store experience tuple (s1, a, r, s2, done) to the memory
            agent.add_experience(s1, action, reward, s2, done)

            # update main q-network
            if iteration % update_main_frequency == 0:
                if len(agent.memory) >= agent.batch_size:
                    agent.update_main_qnetwork()

            # update target q-network
            if iteration % update_target_frequency == 0:
                agent.update_target_qnetwork()

            # update counter
            iteration += 1

            s1 = s2

            if done:
                break

        scores.append(score)
        scores_window.append(score)

        print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(scores_window)), end="\r")
        if episode % 100 == 0:
            print(f"Episode={episode:03}\tAverage Score={np.mean(scores_window)}")

        if np.mean(scores_window) >= TARGET_SCORE:
            print(f"Agent reached target score in {episode} episodes! (Avg. Score = {np.mean(scores_window)})")
            # save main qnetwork
            torch.save(agent.qnetwork.state_dict(), 'checkpoint.pth')
            break

    return scores

import numpy as np
import torch
from collections import namedtuple, deque
import random
from QNetwork import QNetwork


class Agent:

    def __init__(self, env, gamma, buffer_size, batch_size, learning_rate=5e-4, load_weights=None, train_mode=True):

        seed = 110
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.train_mode = train_mode

        # General
        brain_name = env.brain_names[0]
        brain = env.brains[brain_name]
        env_info = env.reset(train_mode=train_mode)[brain_name]
        state = env_info.vector_observations[0]
        self.n_states = len(state)
        self.n_actions = brain.vector_action_space_size

        # Learning
        self.gamma = gamma

        # Q-networks
        self.learning_rate = learning_rate
        self.qnetwork = QNetwork(self.n_states, self.n_actions, seed).to(self.device)
        self.target_qnetwork = QNetwork(self.n_states, self.n_actions, seed).to(self.device)
        self.target_qnetwork.load_state_dict(self.qnetwork.state_dict())
        self.optimizer = torch.optim.Adam(self.qnetwork.parameters(), lr=learning_rate)

        # Experience replay
        self.memory = deque(maxlen=buffer_size)
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.experience = namedtuple('experience', ['state', 'action', 'reward', 'next_state', 'done'])

    def load_weights(self, file_path):
        self.target_qnetwork.load_state_dict(torch.load(file_path))
        self.qnetwork.load_state_dict(torch.load(file_path))

    def update_target_qnetwork(self):
        """
        Copy the weights of the main network to the target network.
        """
        # self.target_qnetwork.set_weights(self.qnetwork.get_weights())
        # alternative solution is to use soft-update:
        # θ_target = τ*θ_local + (1 - τ)*θ_target
        tau = 1e-3
        for target_param, local_param in zip(self.target_qnetwork.parameters(), self.qnetwork.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def update_main_qnetwork(self):
        """
        Updates the main q-network
        """
        # sample
        minibatch = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in minibatch if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in minibatch if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in minibatch if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in minibatch if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in minibatch if e is not None]).astype(np.uint8)).float().to(self.device)

        # Get max predicted Q values (for next states) from target model
        q_targets_next = self.target_qnetwork(next_states).detach().max(1)[0].unsqueeze(1)

        # Compute Q targets for current states
        q_targets = rewards + (self.gamma * q_targets_next * (1 - dones))

        # Get expected Q values from local model
        q_expected = self.qnetwork(states).gather(1, actions)

        # Compute loss
        loss = torch.nn.functional.mse_loss(q_expected, q_targets)

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def take_action(self, state, epsilon):
        """
        The greedy action procedure.
        We use the main network to find the action.
        """
        if np.random.rand() <= epsilon:
            action = torch.tensor([np.random.randint(self.n_actions)], device=self.device, dtype=torch.long)
        else:
            # predict using main network
            with torch.no_grad():
                s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                action = self.qnetwork(s).max(1)[1].view(1, 1)

        return action.item()

    def add_experience(self, state, action, reward, next_state, done):
        """
        Add the latest experience tuple
        """
        exp = self.experience(state, action, reward, next_state, done)
        self.memory.append(exp)

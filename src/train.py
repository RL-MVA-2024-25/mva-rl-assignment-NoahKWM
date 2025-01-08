from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient

env = TimeLimit(
    env=HIVPatient(domain_randomization=True), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.


# Libraries
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
from tqdm import tqdm
from copy import deepcopy
import matplotlib.pyplot as plt
from evaluate import evaluate_HIV

# Device
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
print(f"Device used: {DEVICE}")

# DQN
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        return self.model(x)


class ReplayBuffer:
    def __init__(self, capacity, device):
        self.capacity = capacity # capacity of the buffer
        self.data = []
        self.index = 0 # index of the next cell to be filled
        self.device = device
    def append(self, s, a, r, s_, d):
        if len(self.data) < self.capacity:
            self.data.append(None)
        self.data[self.index] = (s, a, r, s_, d)
        self.index = (self.index + 1) % self.capacity
    def sample(self, batch_size):
        batch = random.sample(self.data, batch_size)
        return list(map(lambda x:torch.Tensor(np.array(x)).to(self.device), list(zip(*batch))))
    def __len__(self):
        return len(self.data)


# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!
class ProjectAgent:
    def __init__(self, buffer_size=10000, batch_size=256, lr=1e-3):
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        self.batch_size = batch_size
        self.gamma = 0.98
        self.device = DEVICE

        # epsilon
        self.epsilon_max = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.epsilon_delay = 600
        self.epsilon_step = (self.epsilon_max - self.epsilon_min) / 15000

        # Q-Networks
        self.q_network = QNetwork(self.state_dim, self.action_dim).to(self.device)
        self.target_network = deepcopy(self.q_network)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # Replay Buffer
        self.replay_buffer = ReplayBuffer(buffer_size, DEVICE)

        # Gradient steps
        self.nb_gradient_steps = 5
        self.criterion = nn.SmoothL1Loss()
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Scheduler
        # self.scheduler = optim.lr_scheduler.StepLR(
        #     self.optimizer, step_size=600, gamma=0.9
        # ) 

        # Update target network
        self.update_target_strategy = 'ema' # 'replace'  'ema'
        self.update_target_freq = 200
        self.update_target_tau = 0.0005

    def act(self, observation, use_random=False):
        with torch.no_grad():
            observation = torch.Tensor(observation).unsqueeze(0).to(self.device)
            q_values = self.q_network(observation)
            return torch.argmax(q_values).item()
    
    def save(self, path):
        torch.save(self.q_network.state_dict(), path)
    
    def load(self):
        self.q_network.load_state_dict(torch.load("q_network_best_final.pth", map_location=self.device, weights_only=True))

    def gradient_step(self, double_dqn=False):
        if len(self.replay_buffer) > self.batch_size:
            X, A, R, Y, D = self.replay_buffer.sample(self.batch_size)

            if double_dqn:
                with torch.no_grad():
                    next_actions = self.q_network(Y).argmax(1)
                    QYmax = self.target_network(Y).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            else:
                QYmax = self.target_network(Y).max(1)[0].detach()
            update = torch.addcmul(R, 1 - D, QYmax, value=self.gamma).unsqueeze(1)
            QXA = self.q_network(X).gather(1, A.to(torch.long).unsqueeze(1))
            loss = self.criterion(QXA, update)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # self.scheduler.step()

    def train(self, max_episode):
        episode_rewards = []
        episode_scores = []
        episode = 0
        episode_cum_reward = 0
        state, _ = env.reset()
        epsilon = self.epsilon_max
        step = 0
        best_score = -np.inf

        while episode < max_episode:
            # update epsilon
            if step > self.epsilon_delay:
                epsilon = max(self.epsilon_min, epsilon - self.epsilon_step)
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = self.act(state)
            # step
            next_state, reward, done, trunc, _ = env.step(action)
            self.replay_buffer.append(state, action, reward, next_state, done)
            episode_cum_reward += reward
            # train
            for _ in range(self.nb_gradient_steps):
                self.gradient_step(double_dqn=False)
            # update target network if needed
            if self.update_target_strategy == 'replace':
                if step % self.update_target_freq == 0: 
                    self.target_network.load_state_dict(self.q_network.state_dict())
            if self.update_target_strategy == 'ema':
                target_state_dict = self.target_network.state_dict()
                model_state_dict = self.q_network.state_dict()
                tau = self.update_target_tau
                for key in model_state_dict:
                    target_state_dict[key] = tau*model_state_dict[key] + (1-tau)*target_state_dict[key]
                self.target_network.load_state_dict(target_state_dict)
            # next step
            step += 1
            if done or trunc:
                score = evaluate_HIV(self, nb_episode=1)
                episode_scores.append(score)
                episode += 1
                episode_rewards.append(episode_cum_reward)
                state, _ = env.reset()
                episode_cum_reward = 0
                # Save the best model
                if episode_scores[-1] > best_score:
                    best_score = episode_scores[-1]
                    self.save("q_network_best.pth")
                print("Episode: {} | Score: {:.2f} | Epsilon: {:.2f}".format(episode, episode_scores[-1], epsilon))

            else:
                state = next_state
        return episode_rewards, episode_scores

if __name__ == "__main__":
    agent = ProjectAgent()
    rewards, scores = agent.train(200)

    plt.plot(scores)
    plt.show()
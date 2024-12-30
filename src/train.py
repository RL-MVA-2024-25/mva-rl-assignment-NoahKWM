from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
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

# Device
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
print(f"Device used: {DEVICE}")

# Define the neural network for DQN
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.model(x)

# Replay Buffer to store transitions
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = []
        self.capacity = capacity

    def add(self, state, action, reward, next_state, done):
        """
        Ajoute une transition au buffer.
        """
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)  # Supprime l'élément le plus ancien
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        Retourne un échantillon aléatoire de transitions.
        """
        return random.sample(self.buffer, batch_size)

    def size(self):
        return len(self.buffer)



# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!
class ProjectAgent:
    def __init__(self, state_dim=env.observation_space.shape[0], action_dim=env.action_space.n
, buffer_size=1000, batch_size=64, gamma=0.99, lr=1e-3):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = 1.0  # Initial exploration rate
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        self.device = DEVICE

        # Q-Networks
        self.q_network = QNetwork(state_dim, action_dim).to(self.device)
        self.target_network = QNetwork(state_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        # Replay Buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
    def act(self, observation, use_random=False):
        if use_random or (np.random.rand() < self.epsilon):
            return np.random.choice(self.action_dim)
        observation = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(observation)
        return torch.argmax(q_values).item()

    def save(self, path):
        torch.save(self.q_network.state_dict(), path)


    def load(self):
        self.q_network.load_state_dict(torch.load("q_network.pth", map_location=self.device))

    def train_step(self):
        if self.replay_buffer.size() < self.batch_size:
            return

        # Sample a batch of transitions
        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Compute Q values
        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute target Q values
        with torch.no_grad():
            max_next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + self.gamma * max_next_q_values * (1 - dones)

        # Loss
        loss = nn.MSELoss()(q_values, target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())


# Train the agent
def train():
    """
    Fonction principale pour entraîner l'agent.
    """
    num_episodes = 20
    max_steps_per_episode = 200
    target_update_freq = 10

    # Agent initialisation
    agent = ProjectAgent()

    for episode in tqdm(range(num_episodes), desc="Training Episodes"):
        state, _ = env.reset()
        total_reward = 0

        for step in tqdm(range(max_steps_per_episode), desc=f"Episode {episode + 1}", leave=False):
            # Action selection
            action = agent.act(state, use_random=True)
            # One step
            next_state, reward, done, truncated, _ = env.step(action)
            # Store transition in the replay buffer
            agent.replay_buffer.add(state, action, reward, next_state, done)
            # Train the agent on a batch
            agent.train_step()
            # Update state
            state = next_state
            total_reward += reward
            # Check if the episode is done
            if done or truncated:
                break

        # Update target network periodically
        if (episode + 1) % target_update_freq == 0:
            agent.update_target_network()

        print(f"Episode {episode + 1}: total reward = {total_reward}")

    # Save the model
    agent.save("q_network.pth")
    print("Model saved")


if __name__ == "__main__":
    train()
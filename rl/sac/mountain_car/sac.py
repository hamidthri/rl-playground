import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.distributions import Normal

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:
    def __init__(self, capacity, state_dim, action_dim):
        self.capacity = capacity
        self.pointer = 0
        self.size = 0
        
        self.state = np.zeros((capacity, state_dim))
        self.action = np.zeros((capacity, action_dim))
        self.reward = np.zeros((capacity, 1))
        self.next_state = np.zeros((capacity, state_dim))
        self.done = np.zeros((capacity, 1))
        
        self.device = device
        
    def add(self, state, action, reward, next_state, done):
        self.state[self.pointer] = state
        self.action[self.pointer] = action
        self.reward[self.pointer] = reward
        self.next_state[self.pointer] = next_state
        self.done[self.pointer] = done
        
        self.pointer = (self.pointer + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        
    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        
        return (
            torch.FloatTensor(self.state[ind]).to(device),
            torch.FloatTensor(self.action[ind]).to(device),
            torch.FloatTensor(self.reward[ind]).to(device),
            torch.FloatTensor(self.next_state[ind]).to(device),
            torch.FloatTensor(self.done[ind]).to(device)
        )
    
    # New method to add priority to successful experiences
    def add_priority_experience(self, state, action, reward, next_state, done, repeat=5):
        """Add important experiences multiple times to increase their sampling probability"""
        for _ in range(repeat):
            self.add(state, action, reward, next_state, done)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, hidden_dim=256):
        super(Actor, self).__init__()
        
        # Increase network capacity for better representation
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),  # Added extra layer
            nn.ReLU()
        )
        
        # Mean and log_std for the Gaussian policy
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        
        self.max_action = max_action
        self.action_dim = action_dim
        
    def forward(self, state):
        x = self.net(state)
        
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, min=-20, max=2)
        std = torch.exp(log_std)
        
        return mean, std
        
    def sample(self, state):
        mean, std = self.forward(state)
        normal = Normal(mean, std)
        
        # Reparameterization trick
        x_t = normal.rsample()
        
        # Compute log probability
        log_prob = normal.log_prob(x_t)
        
        # Apply tanh squashing
        y_t = torch.tanh(x_t)
        action = y_t * self.max_action
        
        # Compute the log probability of the squashed action
        log_prob -= torch.log(self.max_action * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        return action, log_prob
    
    def get_action(self, state, evaluate=False):
        with torch.no_grad():
            if not isinstance(state, torch.Tensor):
                state = torch.FloatTensor(state).to(device).unsqueeze(0)
                
            if evaluate:
                # Use mean action for deterministic evaluation
                mean, _ = self.forward(state)
                return torch.tanh(mean).cpu().numpy().flatten() * self.max_action
            else:
                # Sample from distribution for stochastic action
                action, _ = self.sample(state)
                return action.cpu().numpy().flatten()

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        
        # Increase network capacity for better representation
        # Q1 architecture
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),  # Added extra layer
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Q2 architecture
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),  # Added extra layer
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        
        q1 = self.q1(sa)
        q2 = self.q2(sa)
        
        return q1, q2

class SAC:
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        discount=0.99,
        tau=0.005,
        alpha=0.2,
        policy_freq=2,
        automatic_entropy_tuning=True,
        hidden_dim=256,
        buffer_size=1000000,
        lr=3e-4
    ):
        # Initialize networks with increased hidden dimension
        self.actor = Actor(state_dim, action_dim, max_action, hidden_dim).to(device)
        self.critic = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim, hidden_dim).to(device)
        
        # Copy parameters from critic to target
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
        
        # Initialize optimizers with higher learning rate for faster learning
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size, state_dim, action_dim)
        
        # Set hyperparameters
        self.discount = discount
        self.tau = tau
        self.policy_freq = policy_freq
        self.updates = 0
        
        # Automatic entropy tuning
        self.automatic_entropy_tuning = automatic_entropy_tuning
        if automatic_entropy_tuning:
            self.target_entropy = -torch.prod(torch.Tensor([action_dim]).to(device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
            self.alpha = self.log_alpha.exp()
        else:
            self.alpha = alpha
            
    def select_action(self, state, evaluate=False):
        return self.actor.get_action(state, evaluate=evaluate)
    
    def store_transition(self, state, action, reward, next_state, done):
        # Check if it's a high-reward experience (successful)
        if reward > 10.0:  # Threshold for "valuable" experience
            # Add this experience multiple times to prioritize it
            self.replay_buffer.add_priority_experience(state, action, reward, next_state, done, repeat=10)
        else:
            self.replay_buffer.add(state, action, reward, next_state, done)
    
    def update_parameters(self, batch_size=256):
        # Sample from replay buffer
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)
        
        # Update critic networks
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_state)
            q1_next, q2_next = self.critic_target(next_state, next_action)
            q_next = torch.min(q1_next, q2_next)
            value_target = reward + (1 - done) * self.discount * (q_next - self.alpha * next_log_prob)
        
        # Current Q estimates
        q1, q2 = self.critic(state, action)
        
        # Compute critic loss
        critic_loss = F.mse_loss(q1, value_target) + F.mse_loss(q2, value_target)
        
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)  # Added gradient clipping
        self.critic_optimizer.step()
        
        # Update the actor and alpha (delayed policy update)
        if self.updates % self.policy_freq == 0:
            # Compute actor loss
            new_action, log_prob = self.actor.sample(state)
            q1_new, q2_new = self.critic(state, new_action)
            q_new = torch.min(q1_new, q2_new)
            
            actor_loss = (self.alpha * log_prob - q_new).mean()
            
            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)  # Added gradient clipping
            self.actor_optimizer.step()
            
            # Update automatic entropy tuning parameter
            if self.automatic_entropy_tuning:
                alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
                
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()
                
                self.alpha = self.log_alpha.exp()
            
            # Use a higher value of tau for faster convergence
            tau = min(self.tau * 2, 0.01)  # Increase tau but cap it
            
            # Update target networks
            for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        
        self.updates += 1
        
    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_target.state_dict(), filename + "_critic_target")
        torch.save(self.actor.state_dict(), filename + "_actor")
        
    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_target.load_state_dict(torch.load(filename + "_critic_target"))
        self.actor.load_state_dict(torch.load(filename + "_actor"))
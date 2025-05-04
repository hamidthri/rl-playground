import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ResBlock(nn.Module):
    def __init__(self,
                 Fin,
                 Fout,
                 n_neurons=512):
        super(ResBlock, self).__init__()
        self.Fin = Fin
        self.Fout = Fout
        self.fc1 = nn.Linear(Fin, n_neurons)
        nn.init.uniform_(self.fc1.weight, -1 / math.sqrt(Fin), 1 / math.sqrt(Fin))
        self.bn1 = nn.BatchNorm1d(n_neurons)
        self.fc2 = nn.Linear(n_neurons, Fout)
        nn.init.uniform_(self.fc2.weight, -1 / math.sqrt(n_neurons), 1 / math.sqrt(n_neurons))
        self.bn2 = nn.BatchNorm1d(Fout)
        if Fin != Fout:
            self.fc3 = nn.Linear(Fin, Fout)
        self.ll = nn.LeakyReLU(negative_slope=0.2)
    
    def forward(self, x, final_nl=True):
        Xin = x if self.Fin == self.Fout else self.ll(self.fc3(x))
        Xout = self.fc1(x)  # n_neurons
        # Xout = self.bn1(Xout)
        Xout = self.ll(Xout)
        Xout = self.fc2(Xout)
        # Xout = self.bn2(Xout)
        Xout = Xin + Xout
        if final_nl:
            return self.ll(Xout)
        return Xout


class NetActor(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 n_neurons=512,
                 **kwargs):
        super(NetActor, self).__init__()
        self.bn1 = nn.BatchNorm1d(in_dim)
        self.rb1 = ResBlock(in_dim, in_dim)
        self.rb2 = ResBlock(in_dim + in_dim, in_dim + in_dim)
        # self.rb3 = ResBlock(n_neurons + in_dim, n_neurons)
        self.out1 = nn.Linear(in_dim + in_dim, out_dim - 1)
        nn.init.uniform_(self.out1.weight, -1 / math.sqrt(in_dim), 1 / math.sqrt(in_dim))
        self.out2 = nn.Linear(in_dim + in_dim, out_dim - 1)
        nn.init.uniform_(self.out2.weight, -1 / math.sqrt(in_dim + in_dim), 1 / math.sqrt(in_dim + in_dim))
        self.do = nn.Dropout(p=.1, inplace=False)
    
    def forward(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)
        obs = obs.to(device)
        X0 = obs
        # X0 = self.bn1(X)
        X = self.rb1(X0, True)
        X = self.rb2(torch.cat([X0, X], dim=-1), True)
        # X = self.rb3(torch.cat([X0, X], dim=-1), True)
        output1 = F.sigmoid(self.out1(X))
        output2 = F.tanh(self.out2(X))
        output = torch.cat((output1, output2), -1)
        return output


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_std_init=0.6):
        super(Actor, self).__init__()
        
        # Use the ResNet-based actor network
        self.net_actor = NetActor(state_dim, action_dim + 1)  # +1 because NetActor expects out_dim to be at least 2
        
        # Action standard deviation
        self.action_std = action_std_init
        self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)
    
    def set_action_std(self, new_action_std):
        """Set the standard deviation of the action distribution"""
        self.action_std = new_action_std
        self.action_var = torch.full((self.action_var.shape[0],), new_action_std * new_action_std).to(device)
    
    def forward(self, state):
        """
        Forward pass of the actor
        
        Args:
            state: Current state
            
        Returns:
            action_mean: Mean action value
        """
        # Get output from the network
        output = self.net_actor(state)
        
        # For CartPole, we only need the first value (action_mean)
        # We take just the first element for the action mean (assuming action_dim=1)
        action_mean = output[:, 0].unsqueeze(-1) if len(output.shape) > 1 else output[0].unsqueeze(-1)
        
        return action_mean
    
    def get_dist(self, state):
        """
        Get action distribution for a given state
        
        Args:
            state: Current state
            
        Returns:
            dist: Action distribution
        """
        action_mean = self.forward(state)
        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(device)
        dist = torch.distributions.Normal(action_mean, self.action_std)
        return dist
    
    def act(self, state):
        """
        Sample action from the distribution and compute log probability
        
        Args:
            state: Current state
            
        Returns:
            action: Sampled action
            action_logprob: Log probability of the action
        """
        dist = self.get_dist(state)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        
        return action.detach(), action_logprob.detach()
    
    def evaluate(self, state, action):
        """
        Evaluate action log probability and entropy
        
        Args:
            state: Current state
            action: Action to evaluate
            
        Returns:
            action_logprobs: Log probability of the action
            dist_entropy: Distribution entropy
        """
        dist = self.get_dist(state)
        
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        
        return action_logprobs, dist_entropy
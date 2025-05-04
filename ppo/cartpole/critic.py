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


class NetCritic(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim=1,
                 n_neurons=512,
                 **kwargs):
        super(NetCritic, self).__init__()
        self.bn1 = nn.BatchNorm1d(in_dim)
        self.rb1 = ResBlock(in_dim, in_dim)
        self.rb2 = ResBlock(in_dim + in_dim, in_dim + in_dim)
        # self.rb3 = ResBlock(n_neurons + in_dim, n_neurons)
        self.out = nn.Linear(in_dim + in_dim, out_dim)
        nn.init.uniform_(self.out.weight, -1 / math.sqrt(in_dim + in_dim), 1 / math.sqrt(in_dim + in_dim))
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
        output = self.out(X)
        return output


class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        
        # Use the ResNet-based critic network
        self.net_critic = NetCritic(state_dim, out_dim=1)
    
    def forward(self, state):
        """
        Forward pass through the critic network
        
        Args:
            state: Current state
            
        Returns:
            value: State value estimate
        """
        value = self.net_critic(state)
        return value
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        identity = x
        out = F.relu(self.linear1(x))
        out = self.linear2(out)
        return F.relu(out + identity)

class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.input_layer = nn.Linear(state_dim, hidden_dim)

        # Residual blocks
        self.res_block1 = ResidualBlock(hidden_dim)
        self.res_block2 = ResidualBlock(hidden_dim)

        self.output_layer = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = F.relu(self.input_layer(state))
        x = self.res_block1(x)
        x = self.res_block2(x)
        action_logits = self.output_layer(x)
        return action_logits

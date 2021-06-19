import torch
from torch import nn



class Actor(nn.Module):
     
    def __init__(self, state_dim, n_actions):
        super().__init__()
        self.model = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 32),
                nn.Tanh(),
                nn.Linear(32, n_actions),
                nn.Softmax()
        )

    def forward(self, X):
        return self.model(X)



class Critic(nn.Module):

    def __init__(self, state_dim):
        super().__init__()
        self.model = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
        )

    def forward(self, X):
        return self.model(X)



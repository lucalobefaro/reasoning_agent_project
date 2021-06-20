import torch
from torch import nn



class Actor(nn.Module):
     
    def __init__(self, state_dim, n_actions, n_colors):
        super().__init__()
        self.n_actions = n_actions
        self.model = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 32),
                nn.Tanh(),
                nn.Linear(32, n_actions*n_colors),
                nn.Softmax()
        )

    def forward(self, X):
        X = self.model(X)
        if X[-1] > 0:
            X = X[-self.n_actions:]
        else:
            X = X[:self.n_actions]
        return X

    def save_model_weights(self, path:str):
        torch.save(self.state_dict(), path)

    def load_model_weights(self, path:str, device:str='cpu'):
        self.load_state_dict(torch.load(path, map_location=torch.device(device)))



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

    def save_model_weights(self, path:str):
        torch.save(self.state_dict(), path)

    def load_model_weights(self, path:str, device:str='cpu'):
        self.load_state_dict(torch.load(path, map_location=torch.device(device)))



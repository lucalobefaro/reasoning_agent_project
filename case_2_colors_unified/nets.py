import torch
from torch import nn



class Actor(nn.Module):
     
    def __init__(self, state_dim, n_actions, n_states):
        super().__init__()
        self.shared_net = nn.Sequential(
                        nn.Linear(state_dim, 64),
                        nn.Tanh(),
                        nn.Linear(64, 32),
                        nn.Tanh(),
        )
        self.state_nets = nn.ModuleList([nn.Sequential(
                        nn.Linear(32, n_actions),
                        nn.Softmax(dim=-1)
        ) for i in range(n_states)])
        

    def forward(self, X): 
        # Extract the shared features
        out = self.shared_net(X)
        # Activate the last layers based on the automa state
        return self.state_nets[int(X[-1])](out)
        
    
    def save_model_weights(self, path:str):
        torch.save(self.state_dict(), path)

    def load_model_weights(self, path:str, device:str='cpu'):
        self.load_state_dict(torch.load(path, map_location=torch.device(device)))



class Critic(nn.Module):

    def __init__(self, state_dim, n_states):
        super().__init__()
        self.shared_net = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU()
        )
        self.state_nets = nn.ModuleList([nn.Sequential(
                nn.Linear(32, 1)
        ) for i in range(n_states)])

    def forward(self, X):
        # Extract the common features
        out = self.shared_net(X)
        # Activate the last layers according to autome state
        return self.state_nets[int(X[-1])](out)

    def save_model_weights(self, path:str):
        torch.save(self.state_dict(), path)

    def load_model_weights(self, path:str, device:str='cpu'):
        self.load_state_dict(torch.load(path, map_location=torch.device(device)))

#rwd n passi to goal

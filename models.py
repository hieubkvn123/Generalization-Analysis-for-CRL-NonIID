import torch
import torch.nn as nn

import tqdm
import numpy as np

from norms import frobenius_norm, l21_norm, spectral_norm 
from common import get_default_device, apply_model_to_batch

# Network definition
class Net(nn.Module):
    def __init__(self, in_dim=784, out_dim=64, hidden_dim=128, L=10, device=None):
        super().__init__()
        
        # Store configs
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.L = L

        # Store device
        if device is None:
            self.device = get_default_device()
        else:
            self.device = device
        self.device_type = self.device.type

        # Convert model to own device
        self.to(self.device)
        
        # Create layers
        self.fc_hidden_layers = []
        for _ in range(1, self.L):
            self.fc_hidden_layers.append(
                nn.Linear(hidden_dim, hidden_dim, bias=False)
            )
            self.fc_hidden_layers.append(
                nn.ReLU()    
            )
        self.v = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.ReLU(), 
            *self.fc_hidden_layers
        )
        self.U = nn.Linear(hidden_dim, out_dim)

        # Store reference matrices
        self.references = []
        for l in range(1, self.L + 1):
            self.references.append(self._get_v_layer_weights(layer=l))
        
    def _tensor_to_numpy(self, x):
        if self.device_type == 'cuda':
            return x.cpu().detach().numpy()            
        else:
            return x.detach().numpy()
        
    def forward(self, x):
        return self.U(self.v(x))

def get_model(in_dim=784, out_dim=64, hidden_dim=128, L=10, device=None):
    return Net(in_dim=in_dim, out_dim=out_dim, hidden_dim=hidden_dim, L=L, device=device)

# Define loss functions
def logistic_loss(y, y_positive, y_negatives):
    N, d = y.shape
    h_exp_sum = 0.0
    
    for y_negative in y_negatives:
        h_exp_sum += torch.exp(
            -torch.matmul(
                y.reshape(N, 1, d), 
                (y_positive - y_negative).reshape(N, d, 1)
            ).squeeze(1)
        )
    loss = torch.log(1 + h_exp_sum)
    return loss

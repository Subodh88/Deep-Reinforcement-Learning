import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor_QNetwork(nn.Module):
    """Actor (Policy) Model. Gives actions for a state"""

    def __init__(self, state_size, action_size, fc1_units, fc2_units, use_bn):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor_QNetwork, self).__init__()
        self.use_bn = use_bn
        self.fc1 = nn.Linear(state_size, fc1_units)
        if self.use_bn:
            self.bn1 = nn.BatchNorm1d(fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        if self.use_bn:
            self.bn2 = nn.BatchNorm1d(fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)        
            
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
        self.fc1.bias.data.fill_(0.1)
        self.fc2.bias.data.fill_(0.1)
        self.fc3.bias.data.fill_(0.1)    
    
    def add_parameter_noise(self, scalar_mult):
        self.fc1.weight.data += torch.randn_like(self.fc1.weight.data) * scalar_mult
        self.fc2.weight.data += torch.randn_like(self.fc2.weight.data) * scalar_mult
        self.fc3.weight.data += torch.randn_like(self.fc3.weight.data) * scalar_mult

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        """param state: Input state (Torch Variable : [n,state_dim] )
        return: Output action (Torch Variable: [n,action_dim] )"""
        if self.use_bn:
            x = self.bn1(self.fc1(state))
        else:            
            x = self.fc1(state)
            
        x = F.relu(x)
        x = self.fc2(x)
        if self.use_bn:
            x = self.bn2(x)
        x = F.relu(x)
        x = self.fc3(x)        
        return F.tanh(x)


class Critic_QNetwork(nn.Module):
    """Critic (Value) Model. Predicts Q-value for each (s,a) pair"""

    def __init__(self, state_size, action_size, fc1_units, fc2_units, num_atoms, use_bn, Q_Distribution=False, log_active=False):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
            num_atoms is the granularity of the bins for Q-value distribution probability
        """
        super(Critic_QNetwork, self).__init__()
        self.use_bn = use_bn
        self._Q_Distribution = Q_Distribution
        self._log = log_active
        self.fc1 = nn.Linear(state_size, fc1_units)
        if self.use_bn:
            self.bn1 = nn.BatchNorm1d(fc1_units)
        self.fc2 = nn.Linear(fc1_units+action_size, fc2_units)
        if self.use_bn:
            self.bn2 = nn.BatchNorm1d(fc2_units)
        
        if self._Q_Distribution:
            self.fc3 = nn.Linear(fc2_units, num_atoms)  
        else:
            self.fc3 = nn.Linear(fc2_units, 1)     
        
        self.reset_parameters()
    
    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
        self.fc1.bias.data.fill_(0.1)
        self.fc2.bias.data.fill_(0.1)
        self.fc3.bias.data.fill_(0.1)


    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        """returns Value function Q(s,a) obtained from critic network
        param state: Input state (Torch Variable : [n,state_dim] )
        param action: Input Action (Torch Variable : [n,action_dim] )
        return: Value function : Q(S,a) (Torch Variable : [n,1] )"""
        
        if self.use_bn:
            x = self.bn1(self.fc1(state))
        else:            
            x = self.fc1(state)
            
        x = F.relu(x)
        xs = torch.cat((x, action), dim=1)        
        xs = self.fc2(xs)
        if self.use_bn:
            xs = self.bn2(xs)
        xs = F.relu(xs)
        xs = self.fc3(xs)  
        
        if self._Q_Distribution:
            if self._log:
                return F.log_softmax(xs, dim=-1)
            else:
                return F.softmax(xs, dim=-1) # softmax converts the Q_probs to valid probabilities (i.e. 0 to 1 and all sum to 1)      
        else:
            return xs

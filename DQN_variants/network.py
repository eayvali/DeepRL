# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 23:19:43 2020

@author: elif.ayvali
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class deep_Q_net(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        https://towardsdatascience.com/three-ways-to-build-a-neural-network-in-pytorch-8cea49f9a61a  
        """
        super(deep_Q_net, self).__init__()
        self.seed = torch.manual_seed(seed)
              
        self.dqn_net = nn.Sequential(OrderedDict([
                ('fc1', nn.Linear(state_size, 256)),
                ('relu1', nn.ReLU()),
                ('fc2', nn.Linear(256, 128)),
                ('relu2', nn.ReLU()),
                ('fc3', nn.Linear(128, 64)),
                ('relu3', nn.ReLU()),
                ('fc4', nn.Linear(64, action_size))
                ]))       


    def forward(self, state):
        """Build a network that maps state -> action values."""
        return self.dqn_net(state)



class dueling_Q_net(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            https://lilianweng.github.io/lil-log/2018/05/05/implementing-deep-reinforcement-learning-models.html
        """
        super(dueling_Q_net, self).__init__()
            
        self.feature_modules = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(state_size, 256)),
        ('relu1', nn.ReLU()),
        ('fc2', nn.Linear(256, 128)),
        ('relu2', nn.ReLU()),
        ('fc3', nn.Linear(128, 64)),
        ]))       

        self.value_modules = nn.Sequential(OrderedDict([
        ('fc_v1', nn.Linear(64, 32)),
        ('relu)v1', nn.ReLU()),        
        ('fc_v2', nn.Linear(32, 1)),
        ]))       

        self.advantage_modules = nn.Sequential(OrderedDict([
        ('fc_a1', nn.Linear(64, 32)),
        ('relu_a1', nn.ReLU()),       
        ('fc_a2', nn.Linear(32, action_size)),
        ]))       
    

    def forward(self, state):
        #Get common features
        common_layers=self.feature_modules(state)
        advantage=self.advantage_modules(common_layers)# batch_size x action_size
        value=self.value_modules(common_layers) #batch_size x 1 
        return value + advantage - advantage.mean(dim=1).unsqueeze(1)
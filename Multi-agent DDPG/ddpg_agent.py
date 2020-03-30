# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 15:00:47 2020

@author: elif.ayvali
"""

from ddpg_network import Actor, Critic

import torch
import torch.optim as optim
import numpy as np
import random
import copy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Parameters from OpenAI Baselines
BUFFER_SIZE = int(1e6)   # replay buffer size
BATCH_SIZE = 256         # minibatch size
UPDATE_EVERY = 3         # Number of episodes that should elapse between gradient descent updates
GAMMA = 0.995            # discount factor
TAU = 1e-3               # for soft update of target parameters

LR_ACTOR = 1e-4         # learning rate of the actor 
LR_CRITIC =5e-4        # learning rate of the critic
WEIGHT_DECAY = 0         # L2 weight decay
EPSILON = 1.0            # Exploration noise coefficient
EPSILON_DECAY = 0        # Decay rate for exploration noise
LEARN_TIMES= 2           # Number of times to backprop with the batch
WARM_UP=   0             # Number of steps for uniform-random action selection, before running real policy. Helps exploration.
CLIP_NORM=True





class DDPGAgent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, meta_agent, agent_idx):
        """Initialize an Agent object.
        
        Params
        ======
            meta_agent:
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = meta_agent.state_size
        self.action_size = meta_agent.action_size
        self.num_agents=meta_agent.num_agents
        self.seed = random.seed(meta_agent.random_seed)
        self.agent_idx=agent_idx
        self.epsilon=EPSILON
        print('meta_agent.state_size, action_size,num_agents,seed_agent_idx',meta_agent.state_size,meta_agent.action_size,meta_agent.num_agents,meta_agent.random_seed,agent_idx)

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(self.state_size, self.action_size, meta_agent.random_seed).to(device)
        self.actor_target = Actor(self.state_size, self.action_size, meta_agent.random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)
        self.hard_copy(self.actor_target,self.actor_local)
        # Critic Network (w/ Target Network)
        self.critic_local = Critic(self.state_size*self.num_agents, self.action_size*self.num_agents, meta_agent.random_seed).to(device)
        self.critic_target = Critic(self.state_size*self.num_agents, self.action_size*self.num_agents, meta_agent.random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)  
        self.hard_copy(self.critic_target,self.critic_local)
        print('Agent:',self.agent_idx,'\n Actor-Critic \n',self.actor_local,self.critic_local)
        
        # Noise process
        self.noise = OUNoise(self.action_size, meta_agent.random_seed)
        
               
    def act(self, state, episode_num, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        if episode_num>WARM_UP:
            self.actor_local.eval()    
            with torch.no_grad():
                action = self.actor_local(state).cpu().data.numpy()
            self.actor_local.train()
            if add_noise:
                #print("action_before_noise",action)
                action += self.epsilon * self.noise.sample()
                #print("action_after_noise",action)
        else:
            #generate random action between -1 and 1
            action=np.random.uniform(-1,1,self.action_size)
        return np.clip(action,-1,1)

    def reset(self):
        self.noise.reset()  

    def hard_copy(self,local_model,target_model):
        for target, local in zip(target_model.parameters(), local_model.parameters()):
            target.data.copy_(local.data)
            
class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state















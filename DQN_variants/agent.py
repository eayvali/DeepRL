# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 23:18:57 2020

@author: elif.ayvali
"""
import numpy as np
import random
from collections import namedtuple, deque

from network import deep_Q_net, dueling_Q_net

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e4)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed, learning_alg='vanilla_deep_Q_learning'):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.learning_alg=learning_alg
        
        # Q-Network
        if self.learning_alg=='deep_Q_learning':
            self.qnetwork_local=deep_Q_net(state_size, action_size, seed).to(device)
            self.qnetwork_target=deep_Q_net(state_size, action_size, seed).to(device)

            print('...Running DQN')
        elif self.learning_alg=='double_deep_Q_learning':
            self.qnetwork_local=deep_Q_net(state_size, action_size, seed).to(device)
            self.qnetwork_target=deep_Q_net(state_size, action_size, seed).to(device)
            print('...Running double DQN')
        elif self.learning_alg=='dueling_deep_Q_learning':
            self.qnetwork_local=dueling_Q_net(state_size, action_size, seed).to(device)
            self.qnetwork_target=dueling_Q_net(state_size, action_size, seed).to(device)          
            print('...Running dueling DQN')
        else:
            print('Invalid Algorithm Type')        
        
        print('Network Architecture', self.qnetwork_local)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        # Learn every UPDATE_EVERY time steps:
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample() #returns torch datatype
                self.learn(experiences, GAMMA) #update value parameters

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        #Convert state to torch structure
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        #Evaluate the current network to get action values for the state
        self.qnetwork_local.eval()#This is equivalent with self.train(False)
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        #For windows system, action type should be int32 to play nice with Unity
        if random.random() > eps:
            
            greedy_action=np.argmax(action_values.cpu().data.numpy())
            return greedy_action.astype(np.int32)
        else:
            random_action=random.choice(np.arange(self.action_size))
            return random_action.astype(np.int32)

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor           
        """
        #states: (batchsize x statesize), actions:(batchsize x 1), rewards: (batchsize x 1)
        states, actions, rewards, next_states, dones = experiences
        if self.learning_alg=="deep_Q_learning":
            #qnetwork_target(next_states): Get max predicted Q values (for next states) from target model (batchsize x actionsize)
            #qnetwork_target(next_states).max(1)# (1 x batch size) returns two tensors: max value in each batch(row), the column index at which the max value is found.
            #qnetwork_target(next_states).max(1)[0]) # (1 x batch size)   gets the max value in each batch 
            #qnetwork_target(next_states).max(1)[0].unsqueeze(1)  converts it to (bathsize x 1) 
            #qnetwork_target(next_states).max(1)[0].unsqueeze(1).detach() detaches the output from the computational graph to ensure that these values don’t update the target network when loss.backward() and optimizer.step() are called
            #Q_target weights should not change during learning phase and should be updated periodically by swapping local network weights
            #select greedy actions using target network and use target network to evaluate its q-value
            Q_greedy = self.qnetwork_target(next_states).max(1)[0].unsqueeze(1).detach() #batchsize x 1        
            # ------Compute Q targets for current states------ :        
        elif self.learning_alg=="double_deep_Q_learning" or self.learning_alg=="dueling_deep_Q_learning":
            #select the greedy action using online network
            greedy_actions = self.qnetwork_local(next_states).max(1)[1].unsqueeze(1).detach()#(bathsize x 1) column index at which the max value is found
            #get the q-values of the selected greedy actions using the target network
            Q_greedy = self.qnetwork_target(next_states).gather(1, greedy_actions)   

        Q_target = rewards + (gamma * Q_greedy * (1 - dones))#If it is the last episode (dones=1) only reward is used 

        # -----Get expected Q values from local model------:
        #self.qnetwork_local(states):  batch size x action_size 
        #Get the Q-values for the actions that the agent actually took, gather() function gets this subset  
        Q_est = self.qnetwork_local(states).gather(1, actions)
        # Compute loss
        loss = F.mse_loss(Q_est, Q_target)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # -------Udate target network --_____________-------:
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)             
        
        
    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)#define a new tuple
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
    
    
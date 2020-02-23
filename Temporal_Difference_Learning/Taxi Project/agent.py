# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 18:19:38 2019

@author: elif.ayvali
"""

import numpy as np
from collections import defaultdict


class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        # Learning parameters
        self.alpha = 0.8  # learning rate
        self.gamma = 0.7  # discount factor
        self.epsilon = 0.3  # initial exploration rate
        self.eps_min=0.0001
        self.eps_decay=0.999 
        
    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        # get epsilon-greedy action probabilities
        policy_s = self.__get_epsilon_greedy_probs(self.Q[state])
        # Choose action A_t
        action = np.random.choice(np.arange(self.nA), p=policy_s)               
        return action

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        Q_sa_next=np.max(self.Q[next_state])#q_learning
        self.Q[state][action] = self.__update_Q(self.Q[state][action], Q_sa_next,reward)

    def __get_epsilon_greedy_probs(self, Q_s):
        """ obtains the action probabilities corresponding to epsilon-greedy policy
            Q_s is a (1 x nA) row of possible actions corresponding to the state"""
        self.epsilon = max( self.epsilon*self.eps_decay , self.eps_min)
        policy_s = np.ones(self.nA) * self.epsilon / self.nA
        best_a = np.argmax(Q_s)#return idx of max val
        policy_s[best_a] = 1 - self.epsilon + (self.epsilon / self.nA)
        return policy_s
    
    
    def __update_Q(self,Qsa, Qsa_next, reward):
        """ updates the action-value function estimate using the most recent time step """
        return Qsa + (self.alpha * (reward + (self.gamma * Qsa_next) - Qsa))


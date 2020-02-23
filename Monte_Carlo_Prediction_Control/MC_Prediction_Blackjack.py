# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 15:26:00 2019

Blackjack:
Each state is a 3-tuple of:
    the player's current sum  ∈{0,1,…,31} ,
    the dealer's face up card  ∈{1,…,10} , and
    whether or not the player has a usable ace (no=0 , yes=1 ).
The agent has two potential actions:
    STICK = 0
    HIT = 1

@author: elif.ayvali
"""

import sys
import gym
import numpy as np
from collections import defaultdict
from plot_utils import plot_blackjack_values, plot_policy

def generate_episode_from_limit_stochastic(bj_env):
    #This policy selects action:STICK with 80% probability if the sum is greater than 18;
    # selects action HIT with 80% probability if the sum is 18 or below 
    episode = []
    state = bj_env.reset()
    while True:
        probs = [0.8, 0.2] if state[0] > 18 else [0.2, 0.8]
        action = np.random.choice(np.arange(2), p=probs)
        next_state, reward, done, info = bj_env.step(action)
        episode.append((state, action, reward))
        state = next_state
        if done:
            break
    return episode


def mc_prediction_q(env, num_episodes, generate_episode, gamma=0.9):
    #Every visit MC
    # initialize empty dictionaries of arrays
    returns_sum = defaultdict(lambda: np.zeros(env.action_space.n))
    N = defaultdict(lambda: np.zeros(env.action_space.n))
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    # loop over episodes
    for i_episode in range(1, num_episodes+1):
        # monitor progress
        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()
        episode=generate_episode(env) #(state, action, reward)
        states, actions, rewards=zip(*episode)
        #discount the rewards
        discounts = np.array([gamma**i for i in range(len(rewards)+1)]) 
        # update the sum of the returns, number of visits, and action-value 
        # function estimates for each state-action pair in the episode
        for i, (state, action) in enumerate(zip(states, actions)):
            returns_sum[state][action] += sum(rewards[i:]*discounts[:-(1+i)])
            print(rewards[i:],discounts[:-(1+i)])
            N[state][action] += 1.0
            Q[state][action] = returns_sum[state][action] / N[state][action]     
    return Q

env = gym.make('Blackjack-v0')
# obtain the action-value function
Q = mc_prediction_q(env, 500000, generate_episode_from_limit_stochastic)

# obtain the corresponding state-value function
V_to_plot = dict((k,(k[0]>18)*(np.dot([0.8, 0.2],v)) + (k[0]<=18)*(np.dot([0.2, 0.8],v))) \
         for k, v in Q.items())

# plot the state-value function
plot_blackjack_values(V_to_plot)
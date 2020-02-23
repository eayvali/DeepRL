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
"""

import sys
import gym
import numpy as np
from collections import defaultdict
from plot_utils import plot_blackjack_values, plot_policy

class Run_Tests:
    def test_mc_prediction(env):     
        # obtain the action-value function
        Q = MS_prediction.mc_prediction_q(env, 500000, MS_prediction.generate_episode_from_limit_stochastic)
        
        # obtain the corresponding state-value function
        V_to_plot = dict((k,(k[0]>18)*(np.dot([0.8, 0.2],v)) + (k[0]<=18)*(np.dot([0.2, 0.8],v))) \
                 for k, v in Q.items())
        # plot the state-value function
        plot_blackjack_values(V_to_plot)
        
    def test_mc_control(env):
        # obtain the estimated optimal policy and action-value function
        policy, Q = MC_control.mc_control(env, 500000, 0.02)    
        # obtain the corresponding state-value function
        V = dict((k,np.max(v)) for k, v in Q.items())
        
        # plot the state-value function
        plot_blackjack_values(V)
        # plot the policy
        plot_policy(policy)
        
class MS_prediction:
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
        #Implementation of every visit MC prediction
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

class MC_control:    
    def mc_control(env, num_episodes, alpha, gamma=1.0, eps_start=1.0, eps_decay=.99999, eps_min=0.1):
        """ arguments:
        env: An instance of an OpenAI Gym environment.
        num_episodes:number of episodes that are generated through agent-environment interaction
        alpha: Step-size
        gamma: Discount rate between (0,1](default value: 1).
        outputs:
        Q: A dictionary (of one-dimensional arrays) where Q[s][a] is the estimated action value corresponding to state s and action a.
        policy: A dictionary where policy[s] returns the action that the agent chooses after observing state s."""
        
        nA = env.action_space.n
        Q = defaultdict(lambda: np.zeros(nA))
        epsilon = eps_start
        # loop over episodes
        for i_episode in range(1, num_episodes+1):
            # monitor progress
            if i_episode % 1000 == 0:
                print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
                sys.stdout.flush()
            # set the value of epsilon
            epsilon = max(epsilon*eps_decay, eps_min)
            # generate an episode by following epsilon-greedy policy
            episode = MC_control.generate_episode_from_Q(env, Q, epsilon, nA)
            # update the action-value function estimate using the episode
            Q = MC_control.update_Q(env, episode, Q, alpha, gamma)
        # determine the policy corresponding to the final action-value function estimate
        policy = dict((k,np.argmax(v)) for k, v in Q.items())
        return policy, Q

    def generate_episode_from_Q(env,Q,epsilon,nA):
        """ generates an episode from following the epsilon-greedy policy """
        episode = []
        state = env.reset()
        while True:
            probs=MC_control.get_action_probs(Q[state], epsilon, nA) #Q[state] is the action row corresponding to the state
            action = np.random.choice(np.arange(nA), p=probs) if state in Q else env.action_space.sample()
            next_state, reward, done, info = env.step(action)
            episode.append((state, action, reward))
            state = next_state
            if done:
                break
        return episode


    def get_action_probs(Q_s, epsilon, nA):
        """ obtains the action probabilities corresponding to epsilon-greedy policy
            Q_s is a (1 x nA) row of possible actions corresponding to the state"""
        policy_s = np.ones(nA) * epsilon / nA
        best_a = np.argmax(Q_s)#return idx of max val
        policy_s[best_a] = 1 - epsilon + (epsilon / nA)
        return policy_s
    
    def update_Q(env, episode, Q, alpha, gamma):
        """ updates the action-value function estimate using the most recent episode """
        states, actions, rewards=zip(*episode)
        #discount the rewards
        discounts = np.array([gamma**i for i in range(len(rewards)+1)]) 
        # update the sum of the returns, number of visits, and action-value 
        # function estimates for each state-action pair in the episode
        for i, (state, action) in enumerate(zip(states, actions)):
            Q_prev=Q[state][action]
            #calculate the difference between current estimated and sampled return
            sampled_return=sum(rewards[i:]*discounts[:-(1+i)]) 
            diff_return= sampled_return-Q_prev
            Q[state][action] = Q_prev+alpha*(diff_return)       
        return Q    



env = gym.make('Blackjack-v0')
#run tests
#Run_Tests.test_mc_prediction(env)
Run_Tests.test_mc_control(env)


    
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 01:41:20 2019

@author: elif.ayvali
"""

"""
Gridworld Cliff reinforcement learning task.
The board is a 4x12 matrix, with (using Numpy matrix indexing):
    [3, 0] as the start at bottom-left
    [3, 11] as the goal at bottom-right
    [3, 1..10] as the cliff at bottom-center
Each time step incurs -1 reward, and stepping into the cliff incurs -100 reward 
and a reset to the start. An episode terminates when the agent reaches the goal.

Sarsa, Sarsamax (Q-learning), Expected Sarsa all converge to the optimal action-value function q* 
(and so yield the optimal policy π*) if:
the value of ϵ decays in accordance with the GLIE conditions, and
the step-size parameter α is sufficiently small.

Sarsa and Expected Sarsa are both on-policy TD control algorithms. 
In this case, the same (ϵ-greedy) policy that is evaluated and improved is also used to select actions.
Sarsamax is an off-policy method, where the (greedy) policy that is evaluated and improved is different from the (ϵ-greedy) policy that is used to select actions.
On-policy TD control methods (like Expected Sarsa and Sarsa) have better online performance than off-policy TD control methods (like Sarsamax).
Expected Sarsa generally achieves better performance than Sarsa.
"""

import sys
import gym
import numpy as np
from collections import defaultdict, deque
import matplotlib.pyplot as plt
import check_test
from plot_utils import plot_values

class RunTests:
    def test_sarsa(env):
        # obtain the estimated optimal policy and corresponding action-value function
        Q_sarsa = TD.sarsa(env, 500, .2)    
        # print the estimated optimal policy
        policy_sarsa = np.array([np.argmax(Q_sarsa[key]) if key in Q_sarsa else -1 for key in np.arange(48)]).reshape(4,12)
        check_test.run_check('td_control_check', policy_sarsa)
        print("\nEstimated Optimal Policy (UP = 0, RIGHT = 1, DOWN = 2, LEFT = 3, N/A = -1):")
        print(policy_sarsa)
        
        # plot the estimated optimal state-value function
        V_sarsa = ([np.max(Q_sarsa[key]) if key in Q_sarsa else 0 for key in np.arange(48)])
        plot_values(V_sarsa)
    
    
    def test_q_learning(env):
        #visualize the estimated optimal policy and the corresponding state-value function
        # obtain the estimated optimal policy and corresponding action-value function
        Q_sarsamax = TD.q_learning(env, 500, .2)
        
        # print the estimated optimal policy
        policy_sarsamax = np.array([np.argmax(Q_sarsamax[key]) if key in Q_sarsamax else -1 for key in np.arange(48)]).reshape((4,12))
        check_test.run_check('td_control_check', policy_sarsamax)
        print("\nEstimated Optimal Policy (UP = 0, RIGHT = 1, DOWN = 2, LEFT = 3, N/A = -1):")
        print(policy_sarsamax)
        # plot the estimated optimal state-value function
        plot_values([np.max(Q_sarsamax[key]) if key in Q_sarsamax else 0 for key in np.arange(48)])    
    
    
    def test_expected_sarsa(env):
        # obtain the estimated optimal policy and corresponding action-value function
        Q_expsarsa = TD.expected_sarsa(env, 500, .2)    
        # print the estimated optimal policy
        policy_expsarsa = np.array([np.argmax(Q_expsarsa[key]) if key in Q_expsarsa else -1 for key in np.arange(48)]).reshape(4,12)
        check_test.run_check('td_control_check', policy_expsarsa)
        print("\nEstimated Optimal Policy (UP = 0, RIGHT = 1, DOWN = 2, LEFT = 3, N/A = -1):")
        print(policy_expsarsa)    
        # plot the estimated optimal state-value function
        plot_values([np.max(Q_expsarsa[key]) if key in Q_expsarsa else 0 for key in np.arange(48)])    
    
    
class TD:

    def get_epsilon_greedy_probs(env, Q_s, epsilon):
        """ obtains the action probabilities corresponding to epsilon-greedy policy
            Q_s is a (1 x nA) row of possible actions corresponding to the state"""
        policy_s = np.ones(env.nA) * epsilon / env.nA
        best_a = np.argmax(Q_s)#return idx of max val
        policy_s[best_a] = 1 - epsilon + (epsilon / env.nA)
        return policy_s
    
    def update_Q(Qsa, Qsa_next, reward, alpha, gamma):
        """ updates the action-value function estimate using the most recent time step """
        return Qsa + (alpha * (reward + (gamma * Qsa_next) - Qsa))
    

    def sarsa(env, num_episodes, alpha, gamma=1.0):
    
        '''
        num_episodes: Number of episodes that are generated through agent-environment interaction.
        alpha: Step-size parameter for the update step.
        gamma: Discount rate  between 0 and 1, inclusive (default value: 1).
        Q: Dictionary Q[s][a] represents estimated action value corresponding to state s and action a.
        eps-greedy converges for epsilon = 1/t''' 
        
        #Initialize Q-table    
        Q = defaultdict(lambda: np.zeros(env.nA))
        # initialize performance monitor
        plot_every = 10
        tmp_scores = deque(maxlen=plot_every)
        scores = deque(maxlen=num_episodes)

        # loop over episodes
        for i_episode in range(1, num_episodes+1):
            # monitor progress
            if i_episode % 100 == 0:
                print("\rEpisode {}/{}".format(i_episode, num_episodes), end="")
                sys.stdout.flush()   
            # set the value of epsilon      
            epsilon=0.1
#            epsilon=1/i_episode #use this to pass the check_test

           # initialize score
            score = 0                  
            # begin an episode, observe S0
            state = env.reset()   
            while True:     
                # get epsilon-greedy action probabilities
                policy_s = TD.get_epsilon_greedy_probs(env, Q[state], epsilon)
                # Choose action A_t
                action = np.random.choice(np.arange(env.nA), p=policy_s)
                #Take action A_t, observe R_t+1, S_t+1  
                next_state, reward, done, info = env.step(action)
                # add reward to score
                score += reward
                #Choose action A_t+1 using policy derived from Q (eps_greedy)
                # get epsilon-greedy action probabilities
                policy_s = TD.get_epsilon_greedy_probs(env, Q[next_state], epsilon)
                    #--------------------------------------------------------------#
                #Update TD estimate of Q(S_t,A_t) : this step is unique to sarsa
                next_action = np.random.choice(np.arange(env.nA), p=policy_s)   
                Q_sa_next=Q[next_state][next_action]
                #--------------------------------------------------------------#
                Q[state][action] = TD.update_Q(Q[state][action], Q_sa_next,reward, alpha, gamma) 
                state=next_state
                if done:
                    tmp_scores.append(score)
                    break
            if (i_episode % plot_every == 0):
                scores.append(np.mean(tmp_scores))
        # plot performance
        plt.plot(np.linspace(0,num_episodes,len(scores),endpoint=False),np.asarray(scores))
        plt.xlabel('Episode Number')
        plt.ylabel('Average Reward (Over Next %d Episodes)' % plot_every)
        plt.show()
        # print best 100-episode performance
        print(('Best Average Reward over %d Episodes: ' % plot_every), np.max(scores))    
        return Q
    
    
    def q_learning(env, num_episodes, alpha, gamma=1.0):
        
        '''
        num_episodes: Number of episodes that are generated through agent-environment interaction.
        alpha: Step-size parameter for the update step.
        gamma: Discount rate  between 0 and 1, inclusive (default value: 1).
        Q: Dictionary Q[s][a] represents estimated action value corresponding to state s and action a.'''
        
        #Initialize Q-table    
        Q = defaultdict(lambda: np.zeros(env.nA))
        # initialize performance monitor
        plot_every = 10
        tmp_scores = deque(maxlen=plot_every)
        scores = deque(maxlen=num_episodes)

        # loop over episodes
        for i_episode in range(1, num_episodes+1):
            # monitor progress
            if i_episode % 100 == 0:
                print("\rEpisode {}/{}".format(i_episode, num_episodes), end="")
                sys.stdout.flush()   
            # set the value of epsilon      
            epsilon=0.1
#            epsilon=1.0/i_episode #use this to pass the check_test
           # initialize score
            score = 0                  
            # begin an episode, observe S0
            state = env.reset()   
            while True:     
                # get epsilon-greedy action probabilities
                policy_s = TD.get_epsilon_greedy_probs(env, Q[state], epsilon)
                # Choose action A_t
                action = np.random.choice(np.arange(env.nA), p=policy_s)
                #Take action A_t, observe R_t+1, S_t+1  
                next_state, reward, done, info = env.step(action)
                # add reward to score
                score += reward
                #--------------------------------------------------------------#
                #Update TD estimate of Q(S_t,A_t) : this step is unique to sarsa_max(q_learning)
                Q_sa_next=np.max(Q[next_state]) #max_a Q(S_t+1,a)
                #--------------------------------------------------------------#
                Q[state][action] = TD.update_Q(Q[state][action], Q_sa_next,reward, alpha, gamma)  
                state=next_state
                if done:
                    tmp_scores.append(score)
                    break
            if (i_episode % plot_every == 0):
                scores.append(np.mean(tmp_scores))
        # plot performance
        plt.plot(np.linspace(0,num_episodes,len(scores),endpoint=False),np.asarray(scores))
        plt.xlabel('Episode Number')
        plt.ylabel('Average Reward (Over Next %d Episodes)' % plot_every)
        plt.show()
        # print best 100-episode performance
        print(('Best Average Reward over %d Episodes: ' % plot_every), np.max(scores))    
        return Q
    
    def expected_sarsa(env, num_episodes, alpha, gamma=1.0):
        '''
        num_episodes: Number of episodes that are generated through agent-environment interaction.
        alpha: Step-size parameter for the update step.
        gamma: Discount rate  between 0 and 1, inclusive (default value: 1).
        Q: Dictionary Q[s][a] represents estimated action value corresponding to state s and action a.
        This algorithm eliminates variance in A_t+1 '''
        
        #Initialize Q-table    
        Q = defaultdict(lambda: np.zeros(env.nA))
        # initialize performance monitor
        plot_every = 10
        tmp_scores = deque(maxlen=plot_every)
        scores = deque(maxlen=num_episodes)

        # loop over episodes
        for i_episode in range(1, num_episodes+1):
            # monitor progress
            if i_episode % 100 == 0:
                print("\rEpisode {}/{}".format(i_episode, num_episodes), end="")
                sys.stdout.flush()   
            # set the value of epsilon      
            epsilon=0.005 #use this to pass the check_test
            #epsilon=1.0/i_episode 
           # initialize score
            score = 0                  
            # begin an episode, observe S0
            state = env.reset()   
            while True:     
                # get epsilon-greedy action probabilities
                policy_s = TD.get_epsilon_greedy_probs(env, Q[state], epsilon)
                # Choose action A_t
                action = np.random.choice(np.arange(env.nA), p=policy_s)
                #Take action A_t, observe R_t+1, S_t+1  
                next_state, reward, done, info = env.step(action)
                # add reward to score
                score += reward
                #Choose action A_t+1 using policy derived from Q (eps_greedy)
                # get epsilon-greedy action probabilities
                policy_s = TD.get_epsilon_greedy_probs(env, Q[next_state], epsilon)
                #--------------------------------------------------------------#
                #Update TD estimate of Q(S_t,A_t) : this step is unique to expected_sarsa
                Q_sa_next=np.dot(Q[next_state],policy_s)  #sum_over_a (action_prob*Q(S_t+1,a))
                #--------------------------------------------------------------#
                Q[state][action] = TD.update_Q(Q[state][action], Q_sa_next,reward, alpha, gamma)  
                state=next_state
                if done:
                    tmp_scores.append(score)
                    break
            if (i_episode % plot_every == 0):
                scores.append(np.mean(tmp_scores))
        # plot performance
        plt.plot(np.linspace(0,num_episodes,len(scores),endpoint=False),np.asarray(scores))
        plt.xlabel('Episode Number')
        plt.ylabel('Average Reward (Over Next %d Episodes)' % plot_every)
        plt.show()
        # print best 100-episode performance
        print(('Best Average Reward over %d Episodes: ' % plot_every), np.max(scores))    
        return Q 

   
env = gym.make('CliffWalking-v0')
RunTests.test_sarsa(env)
RunTests.test_q_learning(env)
RunTests.test_expected_sarsa(env)
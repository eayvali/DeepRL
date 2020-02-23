# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 18:19:02 2019

@author: elif.ayvali
"""

from agent import Agent
from monitor import interact
import gym

env = gym.make('Taxi-v3')
agent = Agent()
avg_rewards, best_avg_reward = interact(env, agent)
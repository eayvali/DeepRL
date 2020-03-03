# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 20:39:28 2020

@author: elif.ayvali
"""

import matplotlib.pyplot as plt
import numpy as np

scores = np.load('DDPG_scores.npy')
scores=np.asarray(scores)
means = np.mean(scores,axis=1)
std  = np.std(scores,axis=1)

plt.figure(figsize=(12,6))
plt.rc('font', size=20)          # controls default text sizes
plt.rc('axes', titlesize=20)     # fontsize of the axes title
plt.rc('axes', labelsize=20)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=20)    # fontsize of the tick labels
plt.rc('ytick', labelsize=20)    # fontsize of the tick labels
plt.rc('legend', fontsize=20)    # legend fontsize
plt.rc('figure', titlesize=20)  # fontsize of the figure title
plt.plot(scores,'.')
plt.plot(means,'k-')
plt.fill_between(range(len(means)), means-std, means+std, color='gray')
plt.grid()
plt.xlabel('Episode')
plt.ylabel('Score')
plt.title('DDPG')


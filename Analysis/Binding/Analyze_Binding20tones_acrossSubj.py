#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  7 11:10:36 2021

@author: ravinderjit
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os



data_loc = '/media/ravinderjit/Data_Drive/Data/EEGdata/MTB/Binding'
pickle_loc = data_loc + '/Pickles/'

fig_loc =  '/media/ravinderjit/Data_Drive/Data/Figures/MTBproj/Binding/'

Subjects = ['S211', 'S259', 'S268', 'S269', 'S270', 'S273',
            'S274','S277','S279','S282', 'S288']

A_epochs = []

#%% Load data

for subject in Subjects:
    with open(os.path.join(pickle_loc,subject+'_Binding.pickle'),'rb') as file:
        [t, conds_save, epochs_save] = pickle.load(file)
        
    
    A_epochs.append(epochs_save)
    
    
  
#%% Get evoked responses
    
A_evkd = np.zeros((len(t),len(Subjects),len(conds_save)))

for sub in range(len(Subjects)):
    for cond in range(len(conds_save)):
        A_evkd[:,sub,cond] = A_epochs[sub][cond].mean(axis=0)
        
#%% Plot Average response across Subjects
conds_comp = [[0,1], [2,4], [3,5]]
labels = ['Onset', 'Incoherent to Coherent', 'Coherent to Incoherent']
    
fig,ax = plt.subplots(3,1,sharex=True)

for jj in range(3):
    cnd1 = conds_comp[jj][0]
    cnd2 = conds_comp[jj][1]
    
    onset12_mean = A_evkd[:,:,cnd1].mean(axis=1)
    onset12_sem = A_evkd[:,:,cnd1].std(axis=1) / np.sqrt(A_evkd.shape[1])
    
    ax[jj].plot(t,onset12_mean,label='12')  
    ax[jj].fill_between(t,onset12_mean - onset12_sem, onset12_mean + onset12_sem,alpha=0.5)
    
    onset20_mean = A_evkd[:,:,cnd2].mean(axis=1)
    onset20_sem = A_evkd[:,:,cnd2].std(axis=1) / np.sqrt(A_evkd.shape[1])
    
    ax[jj].plot(t,onset20_mean,label='20')  
    ax[jj].fill_between(t,onset20_mean - onset20_sem, onset20_mean + onset20_sem,alpha=0.5)
    
    ax[jj].ticklabel_format(axis='y',style='sci',scilimits=(0,0))
    ax[jj].set_title(labels[jj])


ax[0].legend()
ax[2].set_xlabel('Time')
ax[2].set_ylabel('$\mu$V')
fig.suptitle('Average Across Participants')


plt.savefig(os.path.join(fig_loc,'All_12vs20.png'),format='png')
    
    

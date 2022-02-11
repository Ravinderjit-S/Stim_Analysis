#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 15:16:16 2022

@author: ravinderjit
"""



import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import os
import pickle


direct_IAC = '/media/ravinderjit/Data_Drive/Data/EEGdata/BinauralMseq_reCollect/StandardWay/'
data_loc = os.path.abspath('/media/ravinderjit/Data_Drive/Data/EEGdata/DynamicBinaural/Pickles_32_refAvg/')
Subject = 'S211'

fig_path = os.path.abspath('/media/ravinderjit/Data_Drive/Data/Figures/DynBin/')


#%% Load Data

with open(os.path.join(direct_IAC, Subject + '_DynBin_SysFunc' + '.pickle'),'rb') as file:     
    [t_75, IAC_Ht_75, Tot_trials_IAC_75] = pickle.load(file)
        

with open(os.path.join(data_loc, Subject+'_DynBin_SysFunc' + '.pickle'),'rb') as file:     
    [t_70, IAC_Ht_70, ITD_Ht, IAC_Htnf, ITD_Htnf, Tot_trials_IAC_70, Tot_trials_ITD] = pickle.load(file)



#%% Plot Ch Cz

plt.rcParams.update({'font.size':12, 'font.family': 'sans-serif', 'font.sans-serif':['Arial']})

fig, ax = plt.subplots(1)
fig.set_size_inches(4.5,4)
ax.plot(t_70, IAC_Ht_70[31,:],label='70',linewidth=2,color='grey')
ax.plot(t_75, IAC_Ht_75[31,:],label='75',linewidth=2,color='k')
ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
ax.set_xlabel('Time (sec)')
ax.legend()
ax.set_ylabel('Amplitude')
ax.set_xlim([-0.05,1.0])
ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
ax.set_yticks([-2e-3,0,2e-3,4e-3])

plt.savefig(os.path.join(fig_path, 'IAC_70vs75.eps') , format='eps')


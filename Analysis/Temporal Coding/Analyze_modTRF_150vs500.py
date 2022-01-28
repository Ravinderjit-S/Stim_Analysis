#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 12:24:53 2022

@author: ravinderjit
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt


data_loc1 = '/media/ravinderjit/Data_Drive/Data/EEGdata/TemporalCoding/AMmseq_10bits/'
pickle_loc1 = data_loc1 + 'Pickles/'


data_loc2 = '/media/ravinderjit/Data_Drive/Data/EEGdata/TemporalCoding/AMmseq_11bits_500/'
pickle_loc2 = data_loc2 + 'Pickles/'

subject = 'S211'

#%% Load Data

with open(os.path.join(pickle_loc1,subject +'_AMmseq10bits_epochs.pickle'),'rb') as file:
    [Ht_epochs_150, t_epochs_150] = pickle.load(file)
    
    
    
with open(os.path.join(pickle_loc2,subject +'_AMmseq11bits_epochs_500.pickle'),'rb') as file:
    [Ht_epochs_500, t_epochs_500] = pickle.load(file)    
    
    

#%% Plot data

ch = 31 # cz
cz_150 = Ht_epochs_150[ch,:,:].mean(axis=0)
cz_150_sem = Ht_epochs_150[ch,:,:].std(axis=0) / np.sqrt(Ht_epochs_150.shape[1])

cz_500 = Ht_epochs_500[ch,:,:].mean(axis=0)
cz_500_sem = Ht_epochs_500[ch,:,:].std(axis=0) / np.sqrt(Ht_epochs_500.shape[1])


plt.figure()
plt.plot(t_epochs_150*1000,cz_150, color='k',linewidth = 2)
plt.fill_between(t_epochs_150*1000,cz_150-cz_150_sem, cz_150 + cz_150_sem)

plt.plot(t_epochs_500*1000,cz_500, color='tab:green',linewidth = 2)
plt.fill_between(t_epochs_500*1000,cz_500-cz_500_sem, cz_500 + cz_500_sem,color='tab:green',alpha=0.5)

plt.xlim([-100,500])
#plt.xticks([7.3,29, 47, 94, 201, 500, 1000],labels=['7.3','29','47','94','201','500','1000'])
plt.xlabel('Time (msec)', fontsize=12)
plt.ylabel('Amplitude',fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.title('mod-TRF Ch. Cz',fontsize=14)
plt.ticklabel_format(axis='y',style='sci',scilimits=(0,0)) 

#%% Do PCA






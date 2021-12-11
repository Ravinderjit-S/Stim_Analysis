#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 11 16:04:10 2021

@author: ravinderjit
"""

import os
import pickle
import numpy as np
import mne
import matplotlib.pyplot as plt
from scipy.signal import freqz
from sklearn.decomposition import PCA
from sklearn.metrics import explained_variance_score
import scipy.io as sio
from scipy.signal import find_peaks

Subjects = ['S207', 'S228','S236','S238','S239','S246','S247','S250']

dataPassive_loc = '/media/ravinderjit/Data_Drive/Data/EEGdata/TemporalCoding/AMmseq_10bits/'
picklePassive_loc = dataPassive_loc + 'Pickles/'


#%% Load Data from initial mod-trf work

A_Tot_trials = []
A_Ht = []
#A_info_obj = []
#A_ch_picks = []

for sub in range(len(Subjects)):
    subject = Subjects[sub]
    if subject == 'S250':
        subject = 'S250_visit2'
    with open(os.path.join(picklePassive_loc,subject +'_AMmseq10bits.pickle'),'rb') as file:
        [t, Tot_trials, Ht, Htnf, info_obj, ch_picks] = pickle.load(file)
        
    A_Tot_trials.append(Tot_trials)
    
    t1 = np.where(t>=0)[0][0]
    t2 = np.where(t>=0.5)[0][0]
    
    A_Ht.append(Ht[-1,t1:t2])
    #A_info_obj.append(info_obj)
    #A_ch_picks.append(ch_picks)
    

print('Done loading mod_trf proj data ...')
t = t[t1:t2]

#%% Load Data from MTB Project

Subjects2 = ['S078', 'S259','S268', 'S269','S270','S271','S273', 'S274' ,'S277','S279','S281','S282', 'S285','S290']

data_loc = '/media/ravinderjit/Data_Drive/Data/EEGdata/MTB/mTRF/'
pickle_loc = data_loc + 'Pickles/'

for sub in range(len(Subjects2)):
    
    subject = Subjects2[sub]
    
    with open(os.path.join(pickle_loc,subject+'_AMmseq10bits_epochs_cz.pickle'),'rb') as file:
        [Ht_epochs, t_epochs] = pickle.load(file)
        
        
    t1 = np.where(t_epochs>=0)[0][0]
    t2 = np.where(t_epochs>=0.5)[0][0]
        
    A_Tot_trials.append(Ht_epochs.shape[0])
    A_Ht.append(Ht_epochs[:,t1:t2].mean(axis=0))


print('Done Loading from MTB Project')
t_epochs = t_epochs[t1:t2]
Subjects = Subjects + Subjects2

#%% Lets get all mod-TRFs into a numpy array and plot them

Cz_Ht = np.zeros((len(Subjects), len(t)))

for sub in range(len(Subjects)):
    Cz_Ht[sub,:] = A_Ht[sub]
    
plt.figure()
plt.plot(t,Cz_Ht.T)   

#%%

pc = PCA(n_components=6)
pc_sp = pc.fit_transform(Cz_Ht)
pc.explained_variance_ratio_

fig,ax = plt.subplots(2,3)
ax = np.reshape(ax,6)
for c in range(6):
    ax[c].plot(t,pc.components_[c,:])

plt.figure()
plt.plot(t,pc.components_.T)

plt.figure()
plt.scatter(pc_sp[:,0],pc_sp[:,1])

plt.figure()
plt.scatter(pc_sp[:,0],pc_sp[:,2])



plt.figure()
plt.plot(np.abs(pc_sp))









#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 18:06:13 2021

@author: ravinderjit
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import pickle


#%% Load higher rate monaural data

data_loc = '/media/ravinderjit/Data_Drive/Data/EEGdata/TemporalCoding/TMTF/'
pickle_loc = data_loc + 'Pickles/'

Subjects_e = ['E001_1','E002_Visit_1','E003', 'E004_Visit_1', 'E005_Visit_1',
            'E006_Visit_1', 'E007_Visit_1', 'E012_Visit_1', 'E014', 'E016',
            'E022_Visit_1' ]


A_Ht_e = []

for sub in range(len(Subjects_e)):
    subject = Subjects_e[sub]
    with open(os.path.join(pickle_loc,subject+'_TMTF_cz.pickle'),'rb') as file:
       [t, Ht, info_obj, ch_picks] = pickle.load(file)
       
   
    A_Ht_e.append(Ht)
    
t_e = t
    
#put together two responses. They were played with opposite polarity
# so can put together here 
for sub in range(len(A_Ht_e)):
    A_Ht_e[sub] = np.concatenate((A_Ht_e[sub][0], A_Ht_e[sub][1]),axis=0)
    
print('Done loading High rate monaural ...')

#%% Load Second Visit of data collection
data_loc = '/media/ravinderjit/Data_Drive/Data/EEGdata/TemporalCoding/AMmseq_10bits/'
pickle_loc = data_loc + 'Pickles/'

A_Tot_trials = []
A_Ht = []
A_Htnf = []
A_info_obj = []
A_ch_picks = []

A_Ht_epochs = []

Subjects = ['S207','S228','S236','S238','S250'] 


for sub in range(len(Subjects)):
    subject = Subjects[sub]
    if subject == 'S250':
        subject = 'S250_visit2'
    with open(os.path.join(pickle_loc,subject +'_AMmseq10bits.pickle'),'rb') as file:
        [t, Tot_trials, Ht, Htnf, info_obj, ch_picks] = pickle.load(file)
        
    A_Tot_trials.append(Tot_trials)
    A_Ht.append(Ht)
    A_Htnf.append(Htnf)
    A_info_obj.append(info_obj)
    A_ch_picks.append(ch_picks)
    
    with open(os.path.join(pickle_loc,subject +'_AMmseq10bits_epochs.pickle'),'rb') as file:
        [Ht_epochs, t_epochs] = pickle.load(file)
    
    A_Ht_epochs.append(Ht_epochs)
    

print('Done loading 2nd visit ...')

#%% Plot this

fig,ax = plt.subplots(6,2)

for sub in range(len(Subjects)):
    ax[sub,0].plot(t_e,A_Ht_e[sub].mean(axis=0))
    ax[sub,1].plot(t_epochs,A_Ht_epochs[sub][-1,:,:].mean(axis=0))
    
    ax[sub,0].set_xlim(0,0.1)
    ax[sub,1].set_xlim(0,0.1)



















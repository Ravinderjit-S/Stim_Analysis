#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 13 21:27:15 2025

@author: ravinderjit
"""
import os
import pickle
import numpy as np
import mne
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import scipy.io as sio
from scipy.signal import find_peaks
from scipy.stats import ttest_rel


#%% Load mseq
mseq_loc = '/media/ravinderjit/Data_Drive/Data/EEGdata/TemporalCoding/mseqEEG_150_bits10_4096.mat'
Mseq_dat = sio.loadmat(mseq_loc)
mseq = Mseq_dat['mseqEEG_4096'].astype(float)
        
#%% Subjects

['S207', 'S211', 'S228','S236','S238', 'S250'] 

Subjects = ['S207', 'S211', 'S228','S236','S238','S250','S239','S246','S247']
#Exclude subjects S246 and S247 as they didn't do hard task
Subjects_sd = ['S207', 'S211', 'S228', 'S236', 'S238', 'S239', 'S250'] 


dataPassive_loc = '/media/ravinderjit/Data_Drive/Data/EEGdata/TemporalCoding/AMmseq_10bits/'
picklePassive_loc = dataPassive_loc + 'Pickles/'

dataCount_loc = '/media/ravinderjit/Data_Drive/Data/EEGdata/TemporalCoding/AMmseq_active/'
pickleCount_loc = dataCount_loc + 'Pickles/'

pickle_loc_sd = '/media/ravinderjit/Data_Drive/Data/EEGdata/TemporalCoding/AMmseq_active_harder/Pickles/'

fig_path = os.path.abspath('/media/ravinderjit/Data_Drive/Data/Figures/TemporalCoding/')

#%% Norm factor
#This is the normalization factor to account for the extension of the mseq to the emseq
cof = 150
T = 1/150 #this is the amount each point is extended
fs = 4096
norm_fac = np.floor(fs*T)


#%% Load Passive Data

A_Tot_trials_pass = []
A_Ht_pass = []
A_Htnf_pass = []
A_info_obj_pass = []
A_ch_picks_pass = []

A_Ht_epochs_pass = []

for sub in range(len(Subjects)):
    print('Loading ' + Subjects[sub])
    subject = Subjects[sub]
    if subject == 'S250':
        subject = 'S250_visit2'
    with open(os.path.join(picklePassive_loc,subject +'_AMmseq10bits.pickle'),'rb') as file:
        [t, Tot_trials, Ht, Htnf, info_obj, ch_picks] = pickle.load(file)
        
    Ht = Ht / norm_fac
    Htnf = Htnf / norm_fac
        
    A_Tot_trials_pass.append(Tot_trials)
    A_Ht_pass.append(Ht)
    A_Htnf_pass.append(Htnf)
    A_info_obj_pass.append(info_obj)
    A_ch_picks_pass.append(ch_picks)
    
    with open(os.path.join(picklePassive_loc,subject +'_AMmseq10bits_epochs.pickle'),'rb') as file:
        [Ht_epochs, t_epochs] = pickle.load(file)
        
    Ht_epochs = Ht_epochs / norm_fac
    
    A_Ht_epochs_pass.append(Ht_epochs)

print('Done loading passive ...')

#%% Load Counting Data

A_Tot_trials_count = []
A_Ht_count = []
A_Htnf_count = []
A_info_obj_count = []
A_ch_picks_count = []

A_Ht_epochs_count = []

for sub in range(len(Subjects)):
    subject = Subjects[sub]
    print ('loading ' + subject)
    with open(os.path.join(pickleCount_loc,subject +'_AMmseq10bits_Active.pickle'),'rb') as file:
        [t, Tot_trials, Ht, Htnf, info_obj, ch_picks] = pickle.load(file)
        
    Ht = Ht / norm_fac
    Htnf = Htnf / norm_fac
        
    A_Tot_trials_count.append(Tot_trials)
    A_Ht_count.append(Ht)
    A_Htnf_count.append(Htnf)
    A_info_obj_count.append(info_obj)
    A_ch_picks_count.append(ch_picks)
    
    with open(os.path.join(pickleCount_loc,subject +'_AMmseq10bits_Active_epochs.pickle'),'rb') as file:
        [Ht_epochs, t_epochs] = pickle.load(file)
        
    Ht_epochs = Ht_epochs / norm_fac
        
    A_Ht_epochs_count.append(Ht_epochs)
    
print('Done Loading Counting data')

#%% Load Shift Detect Data

Subjects_sd = ['S207', 'S211', 'S228', 'S236', 'S238', 'S239', 'S250'] 

A_Tot_trials_sd = []
A_Ht_sd = []
A_info_obj_sd = []
A_ch_picks_sd = []

A_Ht_epochs_sd = []

for sub in range(len(Subjects_sd)):
    subject = Subjects_sd[sub]
    print('loading ' + subject)
    with open(os.path.join(pickle_loc_sd,subject +'_AMmseq10bit_Active_harder.pickle'),'rb') as file:
        [t, Tot_trials, Ht, info_obj, ch_picks] = pickle.load(file)
    
    Ht = Ht / norm_fac
    Htnf = Htnf / norm_fac
    
    A_Tot_trials_sd.append(Tot_trials)
    A_Ht_sd.append(Ht)
    A_info_obj_sd.append(info_obj)
    A_ch_picks_sd.append(ch_picks)
    
    with open(os.path.join(pickle_loc_sd,subject +'_AMmseq10bit_Active_harder_epochs.pickle'),'rb') as file:
        [Ht_epochs, t_epochs] = pickle.load(file)
        
    Ht_epochs = Ht_epochs / norm_fac
    
    A_Ht_epochs_sd.append(Ht_epochs)
    
print('Done loading shift detect data')


#%% Plot Ch. Cz

t_epochs *= 1e3 #change to ms    
fig,ax = plt.subplots(nrows=3,ncols=3,sharex=True)
fig.set_size_inches(14,12)
ax = np.reshape(ax,9)

t_0 = np.where(t_epochs>=0)[0][0]

for sub in range(len(Subjects)):
    subject = Subjects[sub]
    
    ch = 31
    
    #if sub !=2:
        #ax[sub].axes.yaxis.set_visible(False)
        
    # if sub < len(Subjects)/2:
    #     ax[sub].axes.xaxis.set_visible(False)
    
    ch_pass_ind = np.where(A_ch_picks_pass[sub] == ch)[0][0]
    ch_count_ind = np.where(A_ch_picks_count[sub] == ch)[0][0]
    
    cz_pass = A_Ht_epochs_pass[sub][ch_pass_ind,:,:] *1e6
    cz_count = A_Ht_epochs_count[sub][ch_count_ind,:,:] * 1e6
    
    cz_pass_sem = cz_pass.std(axis=0) / np.sqrt(cz_pass.shape[0])
    cz_count_sem = cz_count.std(axis=0) / np.sqrt(cz_count.shape[0])
    
    cz_pass = cz_pass.mean(axis=0)
    cz_count = cz_count.mean(axis=0)
    
    cz_pass = cz_pass - cz_pass[t_0] #make time 0 have value of 0
    cz_count = cz_count - cz_count[t_0] #make time 0 have value of 0
    
    ax[sub].plot(t_epochs,cz_pass, label='Passive', color='k',linewidth=2)
    ax[sub].fill_between(t_epochs,cz_pass-cz_pass_sem, cz_pass+cz_pass_sem,color='k',alpha=0.5)
    
    ax[sub].plot(t_epochs,cz_count, label='Count (easy)', color='tab:blue')
    ax[sub].fill_between(t_epochs,cz_count-cz_count_sem, cz_count+cz_count_sem,color='tab:blue',alpha=0.5)
    
    if (subject in Subjects_sd):
        sub_sd = Subjects_sd.index(subject)
        
        ch_sd_ind = np.where(A_ch_picks_sd[sub_sd] == ch)[0][0]
        cz_sd = A_Ht_epochs_sd[sub_sd][ch_sd_ind,:,:] * 1e6
        
        cz_sd_sem = cz_sd.std(axis=0) / np.sqrt(cz_sd.shape[0])
        cz_sd = cz_sd.mean(axis=0)
        
        cz_sd = cz_sd - cz_sd[t_0]
        
        ax[sub].plot(t_epochs,cz_sd, label='Shift Detect (hard)', color='tab:orange')
        ax[sub].fill_between(t_epochs,cz_sd - cz_sd_sem, cz_sd + cz_sd_sem, color='tab:orange',alpha=0.5)
    
    #ax[sub].set_title('S' + str(sub+1))
    ax[sub].set_title('P' + str(sub+1),fontweight='bold',fontsize=16)
    ax[sub].set_xlim([-0.010*1e3,0.5*1e3])
    #ax[sub].set_xlim([-.005,0.050])
    #ax[sub].set_xticks([0,0.050,0.1])
    #ax[sub].set_xticks([0,0.2,0.4])
    
ax[1].legend(fontsize=12)
#ax[2].axes.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
ax[6].set_xlabel('Time (msec)',fontsize=16)
ax[7].set_xlabel('Time (msec)',fontsize=16)
ax[8].set_xlabel('Time (msec)',fontsize=16)
ax[6].set_ylabel('\u03BCV',fontsize=16)
ax[0].set_ylabel('\u03BCV',fontsize=16)
ax[3].set_ylabel('\u03BCV',fontsize=16)
# ax[0].set_yticks([-15,0,15,30])
# ax[1].set_yticks([-20,-10,0,10])
# ax[2].set_yticks([-30,0,30,60])
# ax[3].set_yticks([-40,-20,0,20])
for a_ in ax:
    a_.tick_params(axis='both', labelsize=14)

plt.savefig(os.path.join(fig_path,'ModTRF_active_V2.svg'),format='svg')


t_epochs /= 1e3 #change back to sec



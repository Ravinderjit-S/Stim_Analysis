#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 16:33:02 2021

@author: ravinderjit
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import mne
from anlffr.preproc import find_blinks
from EEGpp import EEGconcatenateFolder
from mne.preprocessing.ssp import compute_proj_epochs
import os
import pickle




nchans = 34;
refchans = ['EXG1','EXG2']

Subjects = ['S211_ABAB','S211_BABA']

data_loc = '/media/ravinderjit/Data_Drive/Data/EEGdata/Binding_20tones/'
exclude = ['EXG3','EXG4','EXG5','EXG6','EXG7','EXG8']; #don't need these extra external channels that are saved
   
subject = Subjects[0]
datapath = os.path.join(data_loc,subject)

data_eegAB,data_evnt = EEGconcatenateFolder(datapath+'/',nchans,refchans,exclude)
data_eegAB.filter(l_freq=1,h_freq=40)

datapath = os.path.join(data_loc,Subjects[1])
data_eegBA, data_evnt = EEGconcatenateFolder(datapath+'/',nchans,refchans,exclude)
data_eegBA.filter(l_freq=1,h_freq=40)

data_eegAB.info['bads'].append('A23')
data_eegBA.info['bads'].append('A23')


#%% Remove Blinks

blinks = find_blinks(data_eegAB,ch_name = ['A1'],thresh = 100e-6, l_trans_bandwidth = 0.5, l_freq =1.0)
blink_epochs = mne.Epochs(data_eegAB,blinks,998,tmin=-0.25,tmax=0.25,proj=False,
                              baseline=(-0.25,0),reject=dict(eeg=500e-6))
Projs = compute_proj_epochs(blink_epochs,n_grad=0,n_mag=0,n_eeg=8,verbose='DEBUG')

ocular_projs = [Projs[0]]

data_eegAB.add_proj(ocular_projs)
data_eegAB.plot_projs_topomap()
data_eegAB.plot(events=blinks,show_options=True)

blinks = find_blinks(data_eegBA,ch_name = ['A1'],thresh = 100e-6, l_trans_bandwidth = 0.5, l_freq =1.0)
blink_epochs = mne.Epochs(data_eegBA,blinks,998,tmin=-0.25,tmax=0.25,proj=False,
                              baseline=(-0.25,0),reject=dict(eeg=500e-6))
Projs = compute_proj_epochs(blink_epochs,n_grad=0,n_mag=0,n_eeg=8,verbose='DEBUG')

ocular_projs = [Projs[0]]

data_eegBA.add_proj(ocular_projs)
data_eegBA.plot_projs_topomap()
data_eegBA.plot(events=blinks,show_options=True)

#%% Plot Data
reject = dict(eeg=100e-6)
epochsAB = mne.Epochs(data_eegAB,data_evnt,1,tmin=-0.4,tmax=4.4,reject = reject, baseline = (-0.2,0.))
evkdAB = epochsAB.average()
evkdAB.plot(picks=[31],titles='ABAB')

reject = dict(eeg=100e-6)
epochsBA = mne.Epochs(data_eegBA,data_evnt,1,tmin=-0.4,tmax=4.4,reject = reject, baseline = (-0.2,0.))
evkdBA = epochsBA.average()
evkdBA.plot(picks=[31],titles='BABA')

#%% Get Data

t = evkdAB.times
dataAB = evkdAB.data[np.arange(32),:]
dataBA = evkdBA.data[np.arange(32),:]

sbp = [4,4]
sbp2 = sbp

fig,axs = plt.subplots(sbp[0],sbp[1],sharex=True,gridspec_kw=None)
for p1 in range(sbp[0]):
    for p2 in range(sbp[1]):
        axs[p1,p2].plot(t,dataAB[p1*sbp[1]+p2,:],color='tab:blue')
        axs[p1,p2].plot(t,dataBA[p1*sbp[1]+p2,:],color='tab:orange')
        axs[p1,p2].set_title(p1*sbp[1]+p2)    



fig,axs = plt.subplots(sbp2[0],sbp2[1],sharex=True,gridspec_kw=None)
for p1 in range(sbp2[0]):
    for p2 in range(sbp2[1]):
        axs[p1,p2].plot(t,dataAB[p1*sbp2[1]+p2+sbp[0]*sbp[1],:],color='tab:blue')
        axs[p1,p2].plot(t,dataBA[p1*sbp2[1]+p2+sbp[0]*sbp[1],:],color='tab:orange')
        axs[p1,p2].set_title(p1*sbp2[1]+p2+sbp[0]*sbp[1])   



#%% Look at just transitions
fs = epochsAB.info['sfreq']
t_1 = np.where(t>=1)[0][0]
t_2 = np.where(t>=2)[0][0]
t_3 = np.where(t>=3)[0][0]
t_g = int(np.round(fs*0.6))

ch = 4

AB = dataAB[:,t_1:t_1+t_g] + dataAB[:,t_3:t_3+t_g] + dataBA[:,t_2:t_2+t_g]
BA = dataBA[:,t_1:t_1+t_g] + dataBA[:,t_3:t_3+t_g] #+ dataAB[:,t_2:t_2+t_g]


BA2 =  dataAB[:,t_2:t_2+t_g]

AB = AB/3
AB = BA2
BA = BA/2

t_transition = np.arange(0,t_g/fs,1/fs)

plt.figure()
plt.plot(t_transition,AB[ch,:])
plt.plot(t_transition,BA[ch,:])
plt.legend(['A->B','B->A'])
plt.title('Avg of 3 transitions Ch. ' + str(ch+1))
        
fig,axs = plt.subplots(sbp[0],sbp[1],sharex=True,gridspec_kw=None)
for p1 in range(sbp[0]):
    for p2 in range(sbp[1]):
        axs[p1,p2].plot(t_transition,AB[p1*sbp[1]+p2,:],color='tab:blue')
        axs[p1,p2].plot(t_transition,BA[p1*sbp[1]+p2,:],color='tab:orange')
        axs[p1,p2].set_title(p1*sbp[1]+p2)    



fig,axs = plt.subplots(sbp2[0],sbp2[1],sharex=True,gridspec_kw=None)
for p1 in range(sbp2[0]):
    for p2 in range(sbp2[1]):
        axs[p1,p2].plot(t_transition,AB[p1*sbp2[1]+p2+sbp[0]*sbp[1],:],color='tab:blue')
        axs[p1,p2].plot(t_transition,BA[p1*sbp2[1]+p2+sbp[0]*sbp[1],:],color='tab:orange')
        axs[p1,p2].set_title(p1*sbp2[1]+p2+sbp[0]*sbp[1])   

    
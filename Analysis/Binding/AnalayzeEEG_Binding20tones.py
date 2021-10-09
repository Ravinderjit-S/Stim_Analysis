#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 15:49:37 2021

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

Subjects = ['S211']

data_loc = '/media/ravinderjit/Data_Drive/Data/EEGdata/MTB/'
exclude = ['EXG3','EXG4','EXG5','EXG6','EXG7','EXG8']; #don't need these extra external channels that are saved
   
subject = Subjects[0]
datapath = os.path.join(data_loc,subject)

data_eeg,data_evnt = EEGconcatenateFolder(datapath+'/',nchans,refchans,exclude)
data_eeg.filter(l_freq=1,h_freq=40)



#%% Remove Blinks

blinks = find_blinks(data_eeg,ch_name = ['A1'],thresh = 100e-6, l_trans_bandwidth = 0.5, l_freq =1.0)
blink_epochs = mne.Epochs(data_eeg,blinks,998,tmin=-0.25,tmax=0.25,proj=False,
                              baseline=(-0.25,0),reject=dict(eeg=500e-6))
Projs = compute_proj_epochs(blink_epochs,n_grad=0,n_mag=0,n_eeg=8,verbose='DEBUG')

ocular_projs = [Projs[0]]

data_eeg.add_proj(ocular_projs)
data_eeg.plot_projs_topomap()
data_eeg.plot(events=blinks,show_options=True)

#%% Plot Data

conds = ['12','20'] #14,18 for S211 from earlier date
reject = dict(eeg=100e-6)
epochs = []
evkd = []

for cnd in range(len(conds)):
    ep_cnd = mne.Epochs(data_eeg,data_evnt,cnd+1,tmin=-0.4,tmax=5.4,reject = reject, baseline = (-0.2,0.))
    epochs.append(ep_cnd)
    evkd.append(ep_cnd.average())
    evkd[cnd].plot(picks=[31],titles=conds[cnd])

#%% Get Data and plot

t = evkd[0].times
data = []
for cnd in range(len(evkd)):
    data.append(evkd[cnd].data[np.arange(32),:])
    

sbp = [4,4]
sbp2 = sbp
colors = ['tab:blue','tab:orange']

fig,axs = plt.subplots(sbp[0],sbp[1],sharex=True,gridspec_kw=None)
for p1 in range(sbp[0]):
    for p2 in range(sbp[1]):
        for cnd in range(len(data)):
            axs[p1,p2].plot(t,data[cnd][p1*sbp[1]+p2,:],color=colors[cnd])
        axs[p1,p2].set_title(p1*sbp[1]+p2)    



fig,axs = plt.subplots(sbp2[0],sbp2[1],sharex=True,gridspec_kw=None)
for p1 in range(sbp2[0]):
    for p2 in range(sbp2[1]):
        for cnd in range(len(data)):
            axs[p1,p2].plot(t,data[cnd][p1*sbp2[1]+p2+sbp[0]*sbp[1],:],color=colors[cnd])
        axs[p1,p2].set_title(p1*sbp2[1]+p2+sbp[0]*sbp[1])   


plt.figure()
plt.plot(t,data[0][31,:]*1e6)
plt.plot(t,data[1][31,:]*1e6)
plt.xlabel('Time (sec)')
plt.ylabel('$\mu$V')
plt.legend(['12','20'])
plt.xlim([-0.1,5.4])

#%% Look at just transitions
fs = evkd[0].info['sfreq']

t_trans=np.zeros(5,dtype=int)
for tt in range(5):
    t_trans[tt] = np.where(t>=tt)[0][0] 

t_g = int(np.round(fs*0.6))

data_trans = []

for cnd in range(len(data)):
    data_cnd = data[cnd]
    data_tt = np.zeros([32,t_g,t_trans.size])
    for tt in range(t_trans.size):
        data_tt[:,:,tt] = data_cnd[:,t_trans[tt]:t_trans[tt]+t_g]
    data_trans.append(data_tt)
        
data_trans_avg = []
for cnd in range(len(data)):
    onset = data_trans[cnd][:,:,0]
    AB = (data_trans[cnd][:,:,1] + data_trans[cnd][:,:,3]) / 2
    BA = (data_trans[cnd][:,:,2] + data_trans[cnd][:,:,4]) / 2
    data_trans_avg.append([onset,AB,BA])
    

t_transition = np.arange(0,t_g/fs,1/fs)

plt.figure()
cond_tt = ['Oneset','AB','BA']
cond_tt = ['Onset','Incoherent to Coherent', 'Coherent to Incoherent']
cnd_tt=2
for cnd in range(len(data)):
    plt.plot(t_transition,data_trans_avg[cnd][cnd_tt][31,:]*1e6,label=conds[cnd])
    plt.ylim(-1,3.0)
plt.legend()
plt.ylabel('uV')
plt.xlabel('sec')
plt.title(cond_tt[cnd_tt])
        
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

    
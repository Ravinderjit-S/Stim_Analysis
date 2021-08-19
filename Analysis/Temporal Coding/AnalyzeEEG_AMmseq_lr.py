#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 13:00:14 2021

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
import sys
sys.path.append(os.path.abspath('../mseqAnalysis/'))
from mseqHelper import mseqXcorr


from sklearn.decomposition import PCA

nchans = 34;
refchans = ['EXG1','EXG2']

mseq_loc = '/media/ravinderjit/Data_Drive/Data/EEGdata/TemporalCoding/mseqEEG_150_bits10_4096.mat'
Mseq_dat = sio.loadmat(mseq_loc)
mseq = Mseq_dat['mseqEEG_4096'].astype(float)

subject = 'S211'
exclude = ['EXG3','EXG4','EXG5','EXG6','EXG7','EXG8']; #don't need these extra external channels that are saved
num_nfs=1


#%% Load data and filer
data_loc = '/media/ravinderjit/Data_Drive/Data/EEGdata/TemporalCoding/AMmseq_lr/'
pickle_loc = data_loc + 'Pickles/'

datapath = os.path.join(data_loc, subject)
data_eeg,data_evnt = EEGconcatenateFolder(datapath+'/',nchans,refchans,exclude)
data_eeg.filter(l_freq=1,h_freq=500)

#%% Remove bad channels
if subject == 'S211':
    data_eeg.info['bads'].append('A23')

#%% Blink Removal

blinks = find_blinks(data_eeg,ch_name = ['A1'],thresh = 100e-6, l_trans_bandwidth = 0.5, l_freq =1.0)
blink_epochs = mne.Epochs(data_eeg,blinks,998,tmin=-0.25,tmax=0.25,proj=False,
                              baseline=(-0.25,0),reject=dict(eeg=500e-6))
Projs = compute_proj_epochs(blink_epochs,n_grad=0,n_mag=0,n_eeg=8,verbose='DEBUG')
ocular_projs = [Projs[0]]

data_eeg.add_proj(ocular_projs)
data_eeg.plot_projs_topomap()
data_eeg.plot()

del blinks, blink_epochs, Projs, ocular_projs

#%% Plot data

fs = data_eeg.info['sfreq']
reject = dict(eeg=1000e-6)
epochs = []
for cond in range(2):
    epochs.append(mne.Epochs(data_eeg,data_evnt,cond+1,tmin=-0.3,tmax=np.ceil(mseq.size/fs)+0.4, reject = reject, baseline=(-0.3,0.)))  
    epochs[cond].average().plot(titles='Monaural')

#%% Extract part of response when stim is on
    
ch_picks = np.arange(32)
remove_chs = [22]

ch_picks = np.delete(ch_picks,remove_chs)

t = epochs[0].times
t1 = np.where(t>=0)[0][0]
t2 = t1 + mseq.size + int(np.round(0.4*fs))
epdat = []
for cond in range(2):
    epdat.append(epochs[cond].get_data()[:,ch_picks,t1:t2].transpose(1,0,2))
    
t = t[t1:t2]
t = np.concatenate((-t[-int(np.round(0.4*fs)):0:-1],t[:-1]))
info_obj = epochs[0].info

#%% Remove epochs with large deflections 

Reject_Thresh = 200e-6
Tot_trials = []
for cond in range(2):
    Peak2Peak = epdat[cond].max(axis=2) - epdat[cond].min(axis=2)
    mask_trials = np.all(Peak2Peak < Reject_Thresh, axis=0)
    print('rejected ' + str(epdat[cond].shape[1] - sum(mask_trials)) + ' trials due to P2P')
    epdat[cond] = epdat[cond][:,mask_trials,:]
    print('Total Trials Left: ' + str(epdat[cond].shape[1]))
    Tot_trials.append(epdat[cond].shape[1])
    plt.figure()
    plt.plot(Peak2Peak.T)
    plt.title(cond)

#%% Correlation analysis
    
Htnf = []
Ht = []
for cond in range(2):
    Ht.append(mseqXcorr(epdat[cond],mseq[0,:]))
    Htnf_nf = []
    for nf in range(num_nfs):
        print('On nf:' + str(nf))
        resp=epdat[cond]
        inv_inds = np.random.permutation(epdat[cond].shape[1])[:round(epdat[cond].shape[1]/2)]
        resp[:,inv_inds,:] = -resp[:,inv_inds,:]
        Ht_nf = mseqXcorr(resp,mseq[0,:])
        Htnf_nf.append(Ht_nf)
    
    Htnf.append(Htnf_nf)
        
#%% Plot Ht
    
if ch_picks.size == 31:
    sbp = [5,3]
    sbp2 = [4,4]
elif ch_picks.size == 32:
    sbp = [4,4]
    sbp2 = [4,4]
elif ch_picks.size == 30:
    sbp = [5,3]
    sbp2 = [5,3]

fig,axs = plt.subplots(sbp[0],sbp[1],sharex=True,gridspec_kw=None)
for p1 in range(sbp[0]):
    for p2 in range(sbp[1]):
        axs[p1,p2].plot(t,Ht[0][p1*sbp[1]+p2,:],color='tab:blue')
        axs[p1,p2].plot(t,Ht[1][p1*sbp[1]+p2,:],color='tab:orange')
        axs[p1,p2].set_title(ch_picks[p1*sbp[1]+p2])    
        axs[p1,p2].set_xlim([0,0.5])
        # for n in range(num_nfs):
        #     axs[p1,p2].plot(t,Htnf[n][p1*sbp[1]+p2,:],color='grey',alpha=0.3)
    
fig.suptitle('Ht ')


fig,axs = plt.subplots(sbp2[0],sbp2[1],sharex=True,gridspec_kw=None)
for p1 in range(sbp2[0]):
    for p2 in range(sbp2[1]):
        axs[p1,p2].plot(t,Ht[0][p1*sbp2[1]+p2+sbp[0]*sbp[1],:],color='tab:blue')
        axs[p1,p2].plot(t,Ht[1][p1*sbp2[1]+p2+sbp[0]*sbp[1],:],color='tab:orange')
        axs[p1,p2].set_title(ch_picks[p1*sbp2[1]+p2+sbp[0]*sbp[1]])   
        axs[p1,p2].set_xlim([0,0.5])
        # for n in range(num_nfs):
        #     axs[p1,p2].plot(t,Htnf[n][p1*sbp[1]+p2,:],color='grey',alpha=0.3)
    
fig.suptitle('Ht')    

#%% Save Data

with open(os.path.join(pickle_loc,subject+'_AMmseq_lr.pickle'),'wb') as file:
    pickle.dump([t, Tot_trials, Ht, Htnf, info_obj, ch_picks],file)








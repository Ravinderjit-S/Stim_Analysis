#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 15:11:41 2021

@author: ravinderjit
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import mne
from anlffr.preproc import find_blinks
from EEGpp import EEGconcatenateFolder
from mne.preprocessing.ssp import compute_proj_epochs
import os
import spectralAnalysis as sa
import scipy.io as sio
from anlffr.spectral import mtplv
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA




def PLV_Coh(X,Y,TW,fs):
    """
    X is the Mseq
    Y is time x trials
    TW is half bandwidth product 
    """
    X = X.squeeze()
    ntaps = 2*TW - 1
    dpss = sp.signal.windows.dpss(X.size,TW,ntaps)
    N = int(2**np.ceil(np.log2(X.size)))
    f = np.arange(0,N)*fs/N
    PLV_taps = np.zeros([N,ntaps])
    Coh_taps = np.zeros([N,ntaps])
    
    for k in range(0,ntaps):
        print('tap:',k+1,'/',ntaps)
        Xf = sp.fft(X *dpss[k,:],axis=0,n=N)
        Yf = sp.fft(Y * dpss[k,:].reshape(dpss.shape[1],1),axis=0,n=N)
        XYf = Xf.reshape(Xf.shape[0],1) * Yf.conj()
        PLV_taps[:,k] = abs(np.mean(XYf / abs(XYf),axis=1))
        Coh_taps[:,k] = abs(np.mean(XYf,axis=1) / np.mean(abs(XYf),axis=1))
        
    PLV = PLV_taps.mean(axis=1)
    Coh = Coh_taps.mean(axis=1)
    return PLV, Coh, f


nchans = 34;
refchans = ['EXG1','EXG2']

data_eeg = [];
data_evnt = [];
  
#data_loc = '/media/ravinderjit/Storage2/EEGdata'
data_loc = '/media/ravinderjit/Data_Drive/Data/EEGdata/CMR_mseqTarget/'
subject = 'S211'
exclude = ['EXG3','EXG4','EXG5','EXG6','EXG7','EXG8']

Mseq_loc = '/media/ravinderjit/Data_Drive/Stim_Analysis/Stimuli/TemporalCoding/Stim_Dev/mseqEEG_80_4096.mat'
Mseq_dat = sio.loadmat(Mseq_loc)
mseq = Mseq_dat['mseqEEG_4096'].astype(float)

datapath = os.path.join(data_loc,subject)

data_eeg,data_evnt = EEGconcatenateFolder(datapath+'/',nchans,refchans,exclude)
data_eeg.filter(l_freq=1,h_freq=110)

#%% Blink Removal
blinks = find_blinks(data_eeg,ch_name = ['A1'],thresh = 100e-6, l_trans_bandwidth = 0.5, l_freq =1.0, h_freq=10)
blink_epochs = mne.Epochs(data_eeg,blinks,998,tmin=-0.25,tmax=0.25,proj=False,
                              baseline=(-0.25,0),reject=dict(eeg=500e-6))
Projs = compute_proj_epochs(blink_epochs,n_grad=0,n_mag=0,n_eeg=8,verbose='DEBUG')
blink_proj = [Projs[0]]

data_eeg.add_proj(blink_proj)
data_eeg.plot_projs_topomap()
data_eeg.plot(events=blinks,show_options=True)

#%% Extract epochs and Plot
labels = ['Coh 0', 'Coh 1']

tmin = -1.0
tmax = 14
baseline = (-0.2,0)

reject =dict(eeg=100e-6)

epochs_1 = mne.Epochs(data_eeg,data_evnt,[1],tmin=tmin,tmax=tmax,
                              baseline=baseline)#,reject=reject)
epochs_2 = mne.Epochs(data_eeg,data_evnt,[2],tmin=tmin,tmax=tmax,
                              baseline=baseline)#,reject=reject)

evkd_1 = epochs_1.average()
evkd_2 = epochs_2.average()

picks = [30,31,4,25,21,8,7,26]
evkd_1.plot(picks=picks,titles=labels[0])
evkd_2.plot(picks=picks,titles=labels[1])

#%% Extract part of response when stim is on
t = epochs_1.times
t1 = np.where(t>=0)[0][0]
t2 = t1 + mseq.size

# coh_0 = epochs_1.get_data(picks=31)
# coh_0 = coh_0[:,0,t1:t2].T
# coh_1 = epochs_2.get_data(picks=31)
# coh_1 = coh_1[:,0,t1:t2].T

coh_0_ = epochs_1.get_data()[:,:32,t1:t2].transpose(1,0,2)
coh_1_ = epochs_2.get_data()[:,:32,t1:t2].transpose(1,0,2)


#%% Remove epochs with large deflections
Peak2Peak = coh_0_.max(axis=2) - coh_0_.min(axis=2) 
mask_trials = np.all(Peak2Peak < 100e-6,axis=0)
coh_0_ = coh_0_[:,mask_trials,:]

Peak2Peak = coh_1_.max(axis=2) - coh_1_.min(axis=2) 
mask_trials = np.all(Peak2Peak < 100e-6,axis=0)
coh_1_ = coh_1_[:,mask_trials,:]

trials_even = np.min([coh_1_.shape[1],coh_0_.shape[1]])
coh_0_ = coh_0_[:,:trials_even,:]
coh_1_ = coh_1_[:,:trials_even,:]

#%% Compare same number of trials
# tot_trials = min(coh_1.shape[1],coh_0.shape[1])
# coh_0 = coh_0[:,0:tot_trials]
# coh_1 = coh_1[:,0:tot_trials]

#%% correlation analysis

Resp_0 = coh_0_.mean(axis=1)
Resp_1 = coh_1_.mean(axis=1)

fs = epochs_1.info['sfreq']
tt = np.arange(0,mseq.size/fs,1/fs)

tend = 0.5
ind_tend = np.where(tt>=tend)[0][0]

Ht_0 = np.zeros(Resp_0.shape)
Ht_1 = np.zeros(Resp_1.shape)
for ch in range(Ht_0.shape[0]):
    Ht_0[ch,:] = np.correlate(Resp_0[ch,:], mseq[0,:],mode='full')[mseq.size-1:]
    Ht_1[ch,:] = np.correlate(Resp_1[ch,:],mseq[0,:],mode='full')[mseq.size-1:]

Ht_0 = Ht_0[:,:ind_tend]
Ht_1 = Ht_1[:,:ind_tend]
tt = tt[:ind_tend]

sbp = [4,4]
fig,axs = plt.subplots(sbp[0],sbp[1],sharex=True,gridspec_kw=None)
for p1 in range(sbp[0]):
    for p2 in range(sbp[1]):
        axs[p1,p2].plot(tt,Ht_1[p1*sbp[1]+p2,:])
        axs[p1,p2].set_title(p1*sbp[1]+p2+1)    
fig.suptitle('Ht1')

fig,axs = plt.subplots(sbp[0],sbp[1],sharex=True,gridspec_kw=None)
for p1 in range(sbp[0]):
    for p2 in range(sbp[1]):
        axs[p1,p2].plot(tt,Ht_1[p1*sbp[1]+p2+sbp[0]*sbp[1],:])
        axs[p1,p2].set_title(p1*sbp[1]+p2+1+sbp[0]*sbp[1])    
fig.suptitle('Ht1')

sbp = [4,4]
fig,axs = plt.subplots(sbp[0],sbp[1],sharex=True,gridspec_kw=None)
for p1 in range(sbp[0]):
    for p2 in range(sbp[1]):
        axs[p1,p2].plot(tt,Ht_0[p1*sbp[1]+p2,:])
        axs[p1,p2].set_title(p1*sbp[1]+p2+1)    
fig.suptitle('Ht0')

fig,axs = plt.subplots(sbp[0],sbp[1],sharex=True,gridspec_kw=None)
for p1 in range(sbp[0]):
    for p2 in range(sbp[1]):
        axs[p1,p2].plot(tt,Ht_0[p1*sbp[1]+p2+sbp[0]*sbp[1],:])
        axs[p1,p2].set_title(p1*sbp[1]+p2+1+sbp[0]*sbp[1])    
fig.suptitle('Ht0')



nfft = 2**np.log2(Ht_0.size)
Hf_0 = sp.fft(Ht_0,n=nfft,axis=1)
Hf_1 = sp.fft(Ht_1,n=nfft,axis=1)
f = np.arange(0,fs,fs/nfft)


plt.figure()
plt.plot(f,np.abs(Hf_0).T,color='r')
plt.plot(f,np.abs(Hf_1).T,color='b')


pca = PCA(n_components=2)
pca.fit(Ht_1)
pca_space = pca.fit_transform(Ht_1.T)

pca2 = PCA(n_components=2)
pca2.fit(Ht_0)
pca_space2 = pca2.fit_transform(Ht_0.T)

plt.figure()
plt.plot(tt,pca_space)

plt.figure()
plt.plot(tt,pca_space2)









#%% Compute Coherence/PLV


# TW = 12
# Fres = (1/t[-1]) * TW * 2
# fs = epochs_1.info['sfreq']

# PLV_coh0, Coh_coh0, f = PLV_Coh(mseq,coh_0,TW,fs)
# PLV_coh1, Coh_coh1, f = PLV_Coh(mseq,coh_1,TW,fs)


# plt.figure()
# plt.plot(f,Coh_coh0,color='b')
# plt.plot(f,Coh_coh1,color='r')

# plt.figure()
# plt.plot(f,PLV_coh0,color='b')
# plt.plot(f,Coh_coh1,color='r')

#del data_eeg, data_evnt, epochs_1, epochs_2, blinks, blink_epochs,evkd_1,evkd_2


# params=dict()
# params['Fs'] = fs
# params['tapers'] = [12,2*12-1]
# params['fpass'] = [1,100]
# params['itc'] = 0

# plvtap_1, f = mtplv(coh_0_,params)
# plvtap_2, f = mtplv(coh_1_,params)

# plt.figure()
# plt.plot(f,plvtap_1.T[:,2],color='b')
# plt.plot(f,plvtap_2.T[:,2],color='r')












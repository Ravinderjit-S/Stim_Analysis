#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 10:21:25 2020

@author: ravinderjit
"""

import numpy as np
import scipy as sp
import pylab as pl
import matplotlib.pyplot as plt
import scipy.io as sio
import mne
from anlffr.preproc import find_blinks
from EEGpp import EEGconcatenateFolder
from mne.preprocessing.ssp import compute_proj_epochs
import os
import pickle

from anlffr.spectral import mtspecraw
from anlffr.spectral import mtplv




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
    Phase_taps = np.zeros([N,ntaps])
    for k in range(0,ntaps):
        print('tap:',k+1,'/',ntaps)
        Xf = sp.fft(X * dpss[k,:],axis=0,n=N)
        Yf = sp.fft(Y * dpss[k,:].reshape(dpss.shape[1],1),axis=0,n=N)
        XYf = Xf.reshape(Xf.shape[0],1).conj() * Yf
        Phase_taps[:,k] = np.unwrap(np.angle(np.mean(XYf/abs(XYf),axis=1)))
        PLV_taps[:,k] = abs(np.mean(XYf / abs(XYf),axis=1))
        Coh_taps[:,k] = abs(np.mean(XYf,axis=1) / np.mean(abs(XYf),axis=1))
        
    PLV = PLV_taps.mean(axis=1)
    Coh = Coh_taps.mean(axis=1)
    Phase = Phase_taps #Phase_taps.mean(axis=1)
    return PLV, Coh, f, Phase



nchans = 34;
refchans = ['EXG1','EXG2']

data_eeg = [];
data_evnt = [];
  

#Mseq_loc = '/media/ravinderjit/Storage2/EEGdata/mseqEEG_40_4096.mat'
Mseq_loc = '/media/ravinderjit/Data_Drive/Data/EEGdata/TemporalCoding/mseqEEG_150_reps1_4096.mat'
Mseq_dat = sio.loadmat(Mseq_loc)
mseq = Mseq_dat['mseqEEG_4096']
mseq1 = mseq.astype(float)

Mseq_loc = '/media/ravinderjit/Data_Drive/Data/EEGdata/TemporalCoding/mseqEEG_150_reps3_4096.mat'
Mseq_dat = sio.loadmat(Mseq_loc)
mseq = Mseq_dat['mseqEEG_4096']
mseq2 = mseq.astype(float)

#data_loc = '/media/ravinderjit/Storage2/EEGdata/'
data_loc = '/media/ravinderjit/Data_Drive/Data/EEGdata/TemporalCoding/AMmseq/'
subject = 'S211_Reps1_3'
exclude = ['EXG3','EXG4','EXG5','EXG6','EXG7','EXG8']; #don't need these extra external channels that are saved

datapath =  os.path.join(data_loc, subject)
# datapath = '/media/ravinderjit/Data_Drive/Data/EEGdata/EFR'
data_eeg,data_evnt = EEGconcatenateFolder(datapath+'/',nchans,refchans,exclude)
data_eeg.filter(l_freq=1,h_freq=200)

#%% Blink Removal
blinks = find_blinks(data_eeg,ch_name = ['A1'],thresh = 100e-6, l_trans_bandwidth = 0.5, l_freq =1.0)
blink_epochs = mne.Epochs(data_eeg,blinks,998,tmin=-0.25,tmax=0.25,proj=False,
                              baseline=(-0.25,0),reject=dict(eeg=500e-6))
Projs = compute_proj_epochs(blink_epochs,n_grad=0,n_mag=0,n_eeg=8,verbose='DEBUG')
blink_proj = Projs[0]

data_eeg.add_proj(blink_proj)
data_eeg.plot_projs_topomap()
data_eeg.plot()

#%% Plot data

reject = None
epoch_data_1 = mne.Epochs(data_eeg, data_evnt, [1], tmin=-0.3, tmax=14,reject=reject, baseline=(-0.2, 0.)) 
evkd_data_1= epoch_data_1.average();
evkd_data_1.plot(picks = [31], titles = 'AMmseq')

epoch_data_2= mne.Epochs(data_eeg, data_evnt, [2], tmin=-0.3, tmax=11,reject=reject, baseline=(-0.2, 0.)) 
evkd_data_2 = epoch_data_2.average();
evkd_data_2.plot(picks = [31], titles = 'FMmseq')

#del data_eeg, data_evnt, evkd_data_1, evkd_data_2 #, 

#%% Plot PSD

#%% Extract part of response when stim is on
ch_picks = np.array([2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,31,32])

t = epoch_data_1.times
t1 = np.where(t>=0)[0][0]
t2 = t1 + mseq1.size
AM_1 = epoch_data_1.get_data()
AM_1 = AM_1[:,ch_picks-1,t1:t2].transpose(1,0,2)
t_AM1 = t[t1:t2]

t = epoch_data_2.times
t1 = np.where(t>=0)[0][0]
t2 = t1 + int( np.round(mseq2.size / 3))
AM_2 = epoch_data_2.get_data()
AM_2 = AM_2[:,ch_picks-1,t1:t2].transpose(1,0,2)
t_AM2 = t[t1:t2]

fs = epoch_data_1.info['sfreq']

#del epoch_data_1, epoch_data_2

# FMdata = epoch_data_FM.get_data(picks=[31])
# FMdata = FMdata.T[:,0,:]
# FMdata = FMdata[t1:t2,:]

#%% Remove epochs with large deflections


Peak2Peak = AM_1.max(axis=2) - AM_1.min(axis=2)
mask_trials = np.all(Peak2Peak <100e-6,axis=0)
AM_1 = AM_1[:,mask_trials,:]

Peak2Peak = AM_2.max(axis=2) - AM_2.min(axis=2)
mask_trials = np.all(Peak2Peak <100e-6,axis=0)
AM_2 = AM_2[:,mask_trials,:]

AM_1 = AM_1[:,:100,:]
AM_2 = AM_2[:,:100,:]

AM_1 = AM_1.mean(axis=1)
AM_2 = AM_2.mean(axis=1)


params = dict()
params['Fs'] = fs
params['tapers'] = [2,2*2-1]
params['fpass'] = [0, 200]
params['itc'] = 0

TW = 12
Fres = (1/t[-1]) * TW * 2

#PLV_AM, Coh_AM, f, Phase_AM = PLV_Coh(mseq2,AMch_2,TW,fs)
# PLV_FM, Coh_FM, f, Phase_FM = PLV_Coh(mseq,FMdata,TW,fs)


#%% Correlation Analysis


Ht_1 = np.zeros(AM_1.shape)
Ht_2 = np.zeros(AM_2.shape)

for ch in range(Ht_1.shape[0]):
    Ht_1[ch,:] = np.correlate(AM_1[ch,:],mseq1[0,:],mode='full')[mseq1.size-1:]
    Ht_2[ch,:] = np.correlate(AM_2[ch,:],mseq2[0,:],mode='full')[mseq2.size-1:]

tt_1 = np.arange(0,Ht_1.shape[1]/fs,1/fs)
tt_2 = np.arange(0,Ht_2.shape[1]/fs,1/fs)

tend = 0.6
ind_tend1 = np.where(tt_1>=tend)[0][0]
ind_tend2 = np.where(tt_2>=tend)[0][0]

Ht_1 = Ht_1[:,:ind_tend1]
Ht_2 = Ht_2[:,:ind_tend2]

nfft = 2**np.log2(Ht_1.shape[1])

Hf_1 = sp.fft(Ht_1,n=nfft,axis=1)
Hf_2 = sp.fft(Ht_2,n=nfft,axis=1)

phase_1 = np.unwrap(np.angle(Hf_1))
phase_2 = np.unwrap(np.angle(Hf_2))

f = np.arange(0,fs,fs/nfft)
t = np.arange(0,Ht_1.shape[1]/fs,1/fs)

sbp = [5,3]

fig,axs = plt.subplots(sbp[0],sbp[1],sharex=True,gridspec_kw=None)
for p1 in range(sbp[0]):
    for p2 in range(sbp[1]):
        axs[p1,p2].plot(t,Ht_1[p1*sbp[1]+p2,:])
        axs[p1,p2].set_title(p1*sbp[1]+p2+1)    
fig.suptitle('Ht1')


fig,axs = plt.subplots(sbp[0],sbp[1],sharex=True,gridspec_kw=None)
for p1 in range(sbp[0]):
    for p2 in range(sbp[1]):
        axs[p1,p2].plot(t,Ht_1[p1*sbp[1]+p2+sbp[0]*sbp[1],:])
        axs[p1,p2].set_title(p1*sbp[1]+p2+1+sbp[0]*sbp[1])    
fig.suptitle('Ht1')

    
fig,axs = plt.subplots(sbp[0],sbp[1],sharex=True,gridspec_kw=None)
for p1 in range(sbp[0]):
    for p2 in range(sbp[1]):
        axs[p1,p2].plot(f,np.abs(Hf_1[p1*sbp[1]+p2,:]))
        axs[p1,p2].set_title(p1*sbp[1]+p2+1)
        axs[p1,p2].set_xlim([0,150])

fig.suptitle('Hf1')

fig,axs = plt.subplots(sbp[0],sbp[1],sharex=True,gridspec_kw=None)
for p1 in range(sbp[0]):
    for p2 in range(sbp[1]):
        axs[p1,p2].plot(f,np.abs(Hf_1[p1*sbp[1]+p2+sbp[0]*sbp[1],:]))
        axs[p1,p2].set_title(p1*sbp[1]+p2+1+sbp[0]*sbp[1])   
        axs[p1,p2].set_xlim([0,150])
fig.suptitle('Hf1')   






fig,axs = plt.subplots(sbp[0],sbp[1],sharex=True,gridspec_kw=None)
for p1 in range(sbp[0]):
    for p2 in range(sbp[1]):
        axs[p1,p2].plot(t,Ht_2[p1*sbp[1]+p2,:])
        axs[p1,p2].set_title(p1*sbp[1]+p2+1)    
fig.suptitle('Ht2')

fig,axs = plt.subplots(sbp[0],sbp[1],sharex=True,gridspec_kw=None)
for p1 in range(sbp[0]):
    for p2 in range(sbp[1]):
        axs[p1,p2].plot(t,Ht_2[p1*sbp[1]+p2+sbp[0]*sbp[1],:])
        axs[p1,p2].set_title(p1*sbp[1]+p2+1+sbp[0]*sbp[1])    
fig.suptitle('Ht2')

    
fig,axs = plt.subplots(sbp[0],sbp[1],sharex=True,gridspec_kw=None)
for p1 in range(sbp[0]):
    for p2 in range(sbp[1]):
        axs[p1,p2].plot(f,np.abs(Hf_2[p1*sbp[1]+p2,:]))
        axs[p1,p2].set_title(p1*sbp[1]+p2+1)
        axs[p1,p2].set_xlim([0,150])
fig.suptitle('Hf2')

fig,axs = plt.subplots(sbp[0],sbp[1],sharex=True,gridspec_kw=None)
for p1 in range(sbp[0]):
    for p2 in range(sbp[1]):
        axs[p1,p2].plot(f,np.abs(Hf_2[p1*sbp[1]+p2+sbp[0]*sbp[1],:]))
        axs[p1,p2].set_title(p1*sbp[1]+p2+1+sbp[0]*sbp[1])   
        axs[p1,p2].set_xlim([0,150])
fig.suptitle('Hf2')   







#%% Noise floors
# Num_noiseFloors = 10
# Cohnf_AM = np.zeros([Coh_AM.shape[0],Num_noiseFloors])
# PLVnf_AM = np.zeros([Coh_AM.shape[0],Num_noiseFloors])
# # Cohnf_FM = np.zeros([Coh_FM.shape[0],Num_noiseFloors])

# for nf in range(0,Num_noiseFloors):
#     print('NF:',nf+1,'/',Num_noiseFloors)
#     order_AM = np.random.permutation(AMdata.shape[1]-1)
#     # order_FM = np.random.permutation(FMdata.shape[1]-1)
#     Y_AM = AMdata[:,order_AM]
#     # Y_FM = FMdata[:,order_AM]
#     Y_AM[:,0:int(np.round(order_AM.size/2))] = -Y_AM[:,0:int(np.round(order_AM.size/2))]
#     # Y_FM[:,0:int(np.round(order_FM.size/2))] = -Y_FM[:,0:int(np.round(order_FM.size/2))]
    
#     PLVn_AM, Cohn_AM, f, Phase_AM_nf = PLV_Coh(mseq,Y_AM,TW,fs)
#     # PLVn_FM, Cohn_FM, f = PLV_Coh(mseq,Y_FM,TW,fs)
    
#     Cohnf_AM[:,nf] = Cohn_AM
#     PLVnf_AM[:,nf] = PLVn_AM
#     # Cohnf_FM[:,nf] = Cohn_FM
    

# plt.figure()
# plt.plot(f,Cohnf_AM,color='grey')
# plt.plot(f,Coh_AM,color='k',linewidth=2)
# plt.title('AM mseq Coherence')
# plt.ylabel('Coherence')
# plt.xlabel('Frequency (Hz)')
# plt.xlim([0, 45])

# plt.figure()
# plt.plot(f,Cohnf_FM,color='grey')
# plt.plot(f,Coh_FM,color='k',linewidth=2)








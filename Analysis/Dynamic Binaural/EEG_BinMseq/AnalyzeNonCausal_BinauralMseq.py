#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 15:55:59 2021

@author: ravinderjit
"""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import scipy.io as sio
import os
import pickle
import numpy as np
import scipy as sp
import mne
from numpy.fft import fft
from numpy.fft import ifft
from scipy.signal import decimate
import random

import sys
sys.path.append(os.path.abspath('../../mseqAnalysis/'))
from mseqHelper import mseqXcorr


from anlffr.preproc import find_blinks
from EEGpp import EEGconcatenateFolder
from mne.preprocessing.ssp import compute_proj_epochs

Subjects = ['S001','S132','S203','S204','S205','S206','S207','S208','S211']
nchans = 34;


#if Subject == 'S203':
#    refchans = ['EXG3', 'EXG4']
#else:    
#    refchans = ['EXG1','EXG2']
refchans = ['EXG1','EXG2']
refchans = None
    
IAC_eeg = [];
IAC_evnt = [];
ITD_eeg = [];
ITD_evnt = []; 

direct_IAC = '/media/ravinderjit/Data_Drive/Data/EEGdata/DynamicBinaural/IACt/'
direct_ITD = '/media/ravinderjit/Data_Drive/Data/EEGdata/DynamicBinaural/ITDt/'
direct_Mseq = '/media/ravinderjit/Data_Drive/Data/EEGdata/DynamicBinaural/Mseq_4096fs_compensated.mat'



#data_loc = os.path.abspath('/media/ravinderjit/Data_Drive/Data/EEGdata/DynamicBinaural/Pickles_32_refAvg/')

Mseq_mat = sio.loadmat(direct_Mseq)
Mseq = Mseq_mat['Mseq_sig'].T
Mseq = Mseq.astype(float)
# Mseq = decimate(Mseq,2,axis=0)
#fix issues due to filtering from downsampling ... data is sampled at 2048
# Mseq[Mseq<0] = -1
# Mseq[Mseq>0] = 1


Subjects = ['S001','S132','S203','S204','S205','S206','S207','S208','S211']
Subject = 'S211'
exclude = ['EXG1','EXG2','EXG3','EXG4','EXG5','EXG6','EXG7','EXG8'];


#%% Load data
# with open(os.path.join(data_loc, Subject+'_DynBin.pickle'),'rb') as f:
#     IAC_epochs, ITD_epochs = pickle.load(f)
    
IAC_eeg,IAC_evnt = EEGconcatenateFolder(direct_IAC+Subject+'/',nchans,refchans,exclude)
ITD_eeg,ITD_evnt = EEGconcatenateFolder(direct_ITD+Subject+'/',nchans,refchans,exclude)
  
IAC_eeg.filter(1,1000)
ITD_eeg.filter(1,1000)  

#%% Blink removal
blinks_IAC = find_blinks(IAC_eeg, ch_name = ['A1'], thresh = 100e-6,  l_trans_bandwidth=0.5, l_freq = 1.0) 
blinks_ITD = find_blinks(ITD_eeg, ch_name = ['A1'], thresh = 100e-6, l_trans_bandwidth=0.5, l_freq =1.0)
scalings = dict(eeg=40e-6)

blinkIAC_epochs = mne.Epochs(IAC_eeg,blinks_IAC,998,tmin=-0.25,tmax=0.25,proj=False,
                          baseline=(-0.25,0),reject=dict(eeg=500e-6))
blinkITD_epochs = mne.Epochs(ITD_eeg,blinks_ITD,998,tmin=-0.25,tmax=0.25,proj=False,
                          baseline=(-0.25,0),reject=dict(eeg=500e-6))
Projs_IAC = compute_proj_epochs(blinkIAC_epochs, n_grad=0,n_mag=0,n_eeg=8,verbose='DEBUG')
Projs_ITD = compute_proj_epochs(blinkITD_epochs, n_grad=0,n_mag=0,n_eeg=8,verbose='DEBUG')

if Subject == 'S211':                     
    eye_projsIAC = [Projs_IAC[0],Projs_IAC[2]]
    eye_projsITD = [Projs_ITD[0],Projs_ITD[2]]
elif Subject == 'S203':
    eye_projsIAC = [Projs_IAC[0],Projs_IAC[1]]
    eye_projsITD = [Projs_ITD[0],Projs_ITD[1]]
elif Subject == 'S204':
    eye_projsIAC = [Projs_IAC[0],Projs_IAC[2]]
    eye_projsITD = [Projs_ITD[0],Projs_ITD[2]]
elif Subject == 'S132':
    eye_projsIAC = [Projs_IAC[0],Projs_IAC[2]]
    eye_projsITD = [Projs_ITD[0],Projs_ITD[2]]
elif Subject == 'S206':
    eye_projsIAC = [Projs_IAC[0],Projs_IAC[1]]
    eye_projsITD = [Projs_ITD[0],Projs_ITD[1]]
elif Subject == 'S207':
    eye_projsIAC = [Projs_IAC[0]]#,Projs_IAC[2]]
    eye_projsITD = [Projs_ITD[0]]#,Projs_ITD[2]]
elif Subject == 'S205':
    eye_projsIAC = [Projs_IAC[0],Projs_IAC[2]]
    eye_projsITD = [Projs_ITD[0],Projs_ITD[2]]
elif Subject == 'S001':
    eye_projsIAC = [Projs_IAC[0],Projs_IAC[2]]
    eye_projsITD = [Projs_ITD[0],Projs_ITD[2]]
elif Subject == 'S208':
    eye_projsIAC = [Projs_IAC[0],Projs_IAC[2]]
    eye_projsITD = [Projs_ITD[0],Projs_ITD[2]]
    
IAC_eeg.add_proj(eye_projsIAC)
IAC_eeg.plot_projs_topomap()
# IAC_eeg.plot(events=blinks_IAC,scalings=scalings,show_options=True,title = 'IACt')
ITD_eeg.add_proj(eye_projsITD)
ITD_eeg.plot_projs_topomap()

#%% Epoch

rand_IAC_pts = np.zeros((300,3),dtype=int)
rand_ITD_pts = np.zeros((300,3),dtype=int)
for r in np.arange(295):
    rand_IAC_pts[r,:] = [int(random.randint(IAC_evnt[0,0],int(IAC_evnt[-1,0]))),int(0),int(2)]
    rand_ITD_pts[r,:] = [int(random.randint(ITD_evnt[0,0],int(ITD_evnt[-1,0]))),int(0),int(2)]
    
IAC_evnt = np.concatenate([IAC_evnt,rand_IAC_pts])
ITD_evnt = np.concatenate([ITD_evnt,rand_ITD_pts])

IAC_epochs = mne.Epochs(IAC_eeg,IAC_evnt,1,tmin=-0.5,tmax=13,proj=True,baseline=(-0.2, 0.),reject=None)
IAC_epochs.average().plot_topo(title='IAC')

ITD_epochs = mne.Epochs(ITD_eeg,ITD_evnt,1,tmin=-0.5,tmax=13,proj=True,baseline=(-0.2, 0.),reject=None)
ITD_epochs.average().plot_topo(title='ITD')

IAC_epochs_noise = mne.Epochs(IAC_eeg,IAC_evnt,2,tmin=-0.5,tmax=13,proj=True,baseline=(-0.2, 0.),reject=None)
IAC_epochs_noise.average().plot_topo(title='IAC random')

ITD_epochs_noise = mne.Epochs(ITD_eeg,ITD_evnt,2,tmin=-0.5,tmax=13,proj=True,baseline=(-0.2, 0.),reject=None)
ITD_epochs_noise.average().plot_topo(title='ITD random')

#%% Del unnecessary stuff
del IAC_eeg, ITD_eeg, IAC_evnt, ITD_evnt
del blinks_IAC, blinks_ITD, Projs_IAC, Projs_ITD
del blinkIAC_epochs, blinkITD_epochs


#%% Extract epochs when stim is on
t = IAC_epochs.times
fs = IAC_epochs.info['sfreq']
t1 = np.where(t>=0)[0][0]
t2 = t1 + Mseq.size #+ int(np.round(0.4*fs))
t = t[t1:t2]
#t = np.concatenate((-t[-int(np.round(0.4*fs)):0:-1],t[:-1]))
ch_picks = np.arange(32)
ch_picks = [30, 31]

IAC_ep = IAC_epochs.get_data()[:,ch_picks,t1:t2].transpose(1,0,2)
ITD_ep = ITD_epochs.get_data()[:,ch_picks,t1:t2].transpose(1,0,2)
IACn_ep = IAC_epochs_noise.get_data()[:,ch_picks,t1:t2].transpose(1,0,2)
ITDn_ep = ITD_epochs_noise.get_data()[:,ch_picks,t1:t2].transpose(1,0,2)

del IAC_epochs,ITD_epochs,IAC_epochs_noise,ITD_epochs_noise

#%% Remove any epochs with large deflections

reject_thresh = 800e-6
Peak2Peak = IAC_ep.max(axis=2) - IAC_ep.min(axis=2)
mask_trials = np.all(Peak2Peak < reject_thresh,axis=0)
print('rejected ' + str(IAC_ep.shape[1] - sum(mask_trials)) + ' IAC trials due to P2P')
IAC_ep = IAC_ep[:,mask_trials,:]
print('Total IAC trials: ' + str(IAC_ep.shape[1]))

Tot_trials_IAC = IAC_ep.shape[1]

Peak2Peak = ITD_ep.max(axis=2) - ITD_ep.min(axis=2)
mask_trials = np.all(Peak2Peak < reject_thresh,axis=0)
print('rejected ' + str(ITD_ep.shape[1] - sum(mask_trials)) + ' ITD trials due to P2P')
ITD_ep = ITD_ep[:,mask_trials,:]
print('Total ITD trials: ' + str(ITD_ep.shape[1]))

Tot_trials_ITD = ITD_ep.shape[1]

Peak2Peak = IACn_ep.max(axis=2) - IACn_ep.min(axis=2)
mask_trials = np.all(Peak2Peak < reject_thresh,axis=0)
print('rejected ' + str(IACn_ep.shape[1] - sum(mask_trials)) + ' IACn trials due to P2P')
IACn_ep = IACn_ep[:,mask_trials,:]
print('Total IACn trials: ' + str(IACn_ep.shape[1]))

Peak2Peak = ITDn_ep.max(axis=2) - ITDn_ep.min(axis=2)
mask_trials = np.all(Peak2Peak < reject_thresh,axis=0)
print('rejected ' + str(ITDn_ep.shape[1] - sum(mask_trials)) + ' ITDn trials due to P2P')
ITDn_ep = ITDn_ep[:,mask_trials,:]
print('Total IACn trials: ' + str(ITDn_ep.shape[1]))

#%% 

A32_IAC = IAC_ep[1,:,:].mean(axis=0)
A32_IAC = A32_IAC - A32_IAC.mean()

A32_ITD = ITD_ep[1,:,:].mean(axis=0)
A32_ITD = A32_ITD - A32_ITD.mean()

A32_IACn = IACn_ep[1,:,:].mean(axis=0)
A32_IACn = A32_IACn - A32_IACn.mean()

A32_ITDn = ITDn_ep[1,:,:].mean(axis=0)
A32_ITDn = A32_ITDn - A32_ITDn.mean()



#t_xcorr = np.concatenate((-t[-int(np.round(0.4*fs)):0:-1],t[:-1]))
t_xcorr = np.concatenate((-t[-1:0:-1],t))
# t_xcorr = np.append(t_xcorr,t[-1]+1/fs)

plt.figure()
plt.plot(t,A32_IAC)
'A32 IAC'

plt.figure()
plt.plot(t,A32_ITD)
'A32 ITD'

out_IACacorr = np.correlate(A32_IAC,A32_IAC,mode='full')
out_ITDacorr = np.correlate(A32_ITD,A32_ITD,mode='full')

out_IACxcorr = np.correlate(A32_IAC,Mseq[:,0],mode='full')
out_ITDxcorr = np.correlate(A32_ITD,Mseq[:,0],mode='full')

plt.figure()
plt.plot(t_xcorr,out_IACacorr)
plt.title('Acorr IAC')

plt.figure()
plt.plot(t_xcorr,out_ITDacorr)
plt.title('Acorr ITD')

plt.figure()
plt.plot(t_xcorr,out_IACxcorr)
plt.title('Xcorr IAC')

plt.figure()
plt.plot(t_xcorr,out_ITDxcorr)
plt.plot(t_xcorr,out_IACxcorr)
plt.title('Xcorr ITD and IAC')



outn_A32_IACn = np.correlate(A32_IACn,A32_IACn,mode='full')
plt.figure()
plt.plot(t_xcorr,outn_A32_IACn)
plt.title('IAC Acorr noise')

outn_A32_ITDn = np.correlate(A32_ITDn,A32_ITDn,mode='full')
plt.figure()
plt.plot(t_xcorr,outn_A32_IACn)
plt.title('ITD Acorr noise')

outn_A32_IACn_mseq = np.correlate(A32_IACn,Mseq[:,0],mode='full')
plt.figure()
plt.plot(t_xcorr,outn_A32_IACn_mseq)
plt.title('IAC Xcorr noise')

outn_A32_ITDn_mseq = np.correlate(A32_ITDn,Mseq[:,0],mode='full')
plt.figure()
plt.plot(t_xcorr,outn_A32_ITDn_mseq)
plt.title('ITD Xcorr noise')

#%% Model response

rep_mseq = np.concatenate((Mseq[int(fs*1.1):,0],np.zeros(int(fs*1.1))))
repAddMseq = Mseq[:,0] + rep_mseq

plt.figure()
plt.plot(t,repAddMseq)

out_3 = np.correlate(repAddMseq,Mseq[:,0],mode='full')

plt.figure()
plt.plot(t_xcorr,out_3)

plt.figure()
plt.plot(rep_mseq)
plt.plot(Mseq[:,0])
plt.figure()
plt.plot(t_xcorr,out_ITDxcorr)
plt.plot(t_xcorr,out_IACxcorr)
plt.title('Xcorr ITD')


mseq_part = Mseq[:int(round(fs*8)),0]
mseq_test = np.concatenate([mseq_part,np.zeros(Mseq.size-mseq_part.size)])
out_4 = np.correlate(A32_IAC,mseq_test,mode='full')
plt.figure()
plt.plot(t_xcorr,out_4)
plt.plot(t_xcorr,out_IACxcorr)
plt.title('testMseq IAC Xcorr')



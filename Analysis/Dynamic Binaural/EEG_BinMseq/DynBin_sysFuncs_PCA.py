#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 09:32:34 2020

@author: ravinderjit

Reject trials with large p2p deflections
Calculate system functions from EEG data for Dynamic Binaural Msequence project
Also compute noise floors
"""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import scipy.io as sio
import os
import pickle
import numpy as np
import scipy as sp
from anlffr.spectral import mtspecraw
from spectralAnalysis import periodogram
from multiprocessing import Pool
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
import mne
from numpy.fft import fft
from numpy.fft import ifft

import sys
sys.path.append(os.path.abspath('../../mseqAnalysis/'))
from mseqHelper import mseqXcorr

# import mne

def periodic_corr(x, y):
    """Periodic correlation, implemented using the FFT.
    x and y must be real sequences with the same length.
    """
    return ifft(fft(x) * fft(y).conj()).real

direct_Mseq = '/media/ravinderjit/Data_Drive/Data/EEGdata/DynamicBinaural/Mseq_4096fs_compensated.mat'
data_loc = os.path.abspath('/media/ravinderjit/Data_Drive/Data/EEGdata/DynamicBinaural/Pickles_32_refAvg/')

Mseq_mat = sio.loadmat(direct_Mseq)
Mseq = Mseq_mat['Mseq_sig'].T
Mseq = Mseq.astype(float)
Mseq = sp.signal.decimate(Mseq,2,axis=0)
#fix issues due to filtering from downsampling ... data is sampled at 2048
Mseq[Mseq<0] = -1
Mseq[Mseq>0] = 1


Num_noiseFloors = 1


Subjects = ['S001','S132','S203','S204','S205','S206','S207','S208','S211']


for subj in range(0,len(Subjects)):
    Subject = Subjects[subj]
    print(Subject, '.........................')
    with open(os.path.join(data_loc, Subject+'_DynBin.pickle'),'rb') as f:
        IAC_epochs, ITD_epochs = pickle.load(f)
    #IAC_evoked = IAC_epochs.average()
    #mne.viz.plot_evoked_topomap(IAC_evoked)
    #%% Extract epochs when stim is on
    t = IAC_epochs.times
    fs = IAC_epochs.info['sfreq']
    t1 = np.where(t>=0)[0][0]
    t2 = t1 + Mseq.size + int(np.round(0.4*fs))
    t = t[t1:t2]
    t = np.concatenate((-t[-int(np.round(0.4*fs)):0:-1],t[:-1]))
    ch_picks = np.arange(32)

    IAC_ep = IAC_epochs.get_data()[:,ch_picks,t1:t2].transpose(1,0,2)
    ITD_ep = ITD_epochs.get_data()[:,ch_picks,t1:t2].transpose(1,0,2)
    
    
    #%% Remove any epochs with large deflections
        
    reject_thresh = 200e-6
    
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
    
        
    #%% Calculate Ht
    
    IAC_Ht = mseqXcorr(IAC_ep,Mseq[:,0])
    ITD_Ht = mseqXcorr(ITD_ep,Mseq[:,0])
    
    
    #%% Calculate Noise Floors
    IAC_Htnf = []
    ITD_Htnf = []
    for nf in range(Num_noiseFloors):
        IAC_nf = IAC_ep.copy()
        ITD_nf = ITD_ep.copy()
        IACrnd_inds = np.random.permutation(IAC_ep.shape[1])[:round(IAC_ep.shape[1]/2)]
        ITDrnd_inds = np.random.permutation(ITD_ep.shape[1])[:round(ITD_ep.shape[1]/2)]
        IAC_nf[:,IACrnd_inds,:] = - IAC_nf[:,IACrnd_inds,:]
        ITD_nf[:,ITDrnd_inds,:] = - ITD_nf[:,ITDrnd_inds,:]

        IAC_Htnf_n = mseqXcorr(IAC_nf,Mseq[:,0])
        ITD_Htnf_n = mseqXcorr(ITD_nf,Mseq[:,0])


        IAC_Htnf.append(IAC_Htnf_n)
        ITD_Htnf.append(ITD_Htnf_n)
    
        
    #%% Save Analysis for subject 
    with open(os.path.join(data_loc, Subject+'_DynBin_SysFunc' + '.pickle'),'wb') as file:     
        pickle.dump([t, IAC_Ht, ITD_Ht, IAC_Htnf, ITD_Htnf, Tot_trials_IAC, Tot_trials_ITD],file)

    del t, IAC_Ht, ITD_Ht, IAC_Htnf, ITD_Htnf, IAC_epochs, ITD_epochs
    
        
        
        
        
    
    
    
    
    
    
    
    
    






#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 09:32:34 2020

@author: ravinderjit

Calculate system functions from EEG data for Dynamic Binaural Msequence project
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


Num_noiseFloors = 100


Subjects = ['S001','S132','S203','S204','S205','S206','S207','S208','S211']
#Something up with S204
Subjects = ['S204']

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

        
    #%% Calculate Ht
    
    #plt.figure()
    # plt.plot(IAC_ep.mean(axis=1))
    # plt.plot(ITD_ep.mean(axis=1))
    
    IAC_Ht = mseqXcorr(IAC_ep,Mseq[:,0])
    ITD_Ht = mseqXcorr(ITD_ep,Mseq[:,0])
    
    
    #%% Calculate Noise Floors
    IAC_Htnf = []
    ITD_Htnf = []
    for nf in range(Num_noiseFloors):
        IAC_nf = IAC_ep
        ITD_nf = ITD_ep
        IACrnd_inds = np.random.permutation(IAC_ep.shape[1])[:round(IAC_ep.shape[1]/2)]
        ITDrnd_inds = np.random.permutation(ITD_ep.shape[1])[:round(ITD_ep.shape[1]/2)]
        IAC_nf[:,IACrnd_inds,:] = - IAC_nf[:,IACrnd_inds,:]
        ITD_nf[:,ITDrnd_inds,:] = - ITD_nf[:,ITDrnd_inds,:]

        IAC_Htnf_n = mseqXcorr(IAC_nf,Mseq[:,0])
        ITD_Htnf_n = mseqXcorr(ITD_nf,Mseq[:,0])

        # IAC_Htnf_n = np.zeros([resp_IAC_nf.shape[0],resp_IAC_nf.shape[1]*2-1])
        # ITD_Htnf_n = np.zeros([resp_ITD_nf.shape[0],resp_ITD_nf.shape[1]*2-1])
        # for ch in ch_picks:
        #     IAC_Htnf_n[ch,:] = np.correlate(resp_IAC_nf[ch,:],Mseq[:,0],mode='full')#[Mseq.size-1:]
        #     ITD_Htnf_n[ch,:] = np.correlate(resp_ITD_nf[ch,:],Mseq[:,0],mode='full')#[Mseq.size-1:]
        # # IAC_Htnf_n = IAC_Htnf_n[:,:tend_ind]
        # ITD_Htnf_n = ITD_Htnf_n[:,:tend_ind]
        IAC_Htnf.append(IAC_Htnf_n)
        ITD_Htnf.append(ITD_Htnf_n)
    
        
    
    #%% Plot Ht
    # num_nfs = Num_noiseFloors
        
    # if ch_picks.size == 31:
    #     sbp = [5,3]
    #     sbp2 = [4,4]
    # elif ch_picks.size == 32:
    #     sbp = [4,4]
    #     sbp2 = [4,4]
    # elif ch_picks.size == 30:
    #     sbp = [5,3]
    #     sbp2 = [5,3]

        
    # fig,axs = plt.subplots(sbp[0],sbp[1],sharex=True)
    # for p1 in range(sbp[0]):
    #     for p2 in range(sbp[1]):
    #         axs[p1,p2].plot(t,IAC_Ht[p1*sbp[1]+p2,:],color='k')
    #         axs[p1,p2].set_title(ch_picks[p1*sbp[1]+p2])    
    #         for n in range(num_nfs):
    #             axs[p1,p2].plot(t,IAC_Htnf[n][p1*sbp[1]+p2,:],color='grey',alpha=0.3)
            
    # fig.suptitle('Ht IAC')
    
    
    # fig,axs = plt.subplots(sbp2[0],sbp2[1],sharex=True,gridspec_kw=None)
    # for p1 in range(sbp2[0]):
    #     for p2 in range(sbp2[1]):
    #         axs[p1,p2].plot(t,IAC_Ht[p1*sbp2[1]+p2+sbp[0]*sbp[1],:],color='k')
    #         axs[p1,p2].set_title(ch_picks[p1*sbp2[1]+p2+sbp[0]*sbp[1]])   
    #         for n in range(num_nfs):
    #             axs[p1,p2].plot(t,IAC_Htnf[n][p1*sbp2[1]+p2+sbp[0]*sbp[1],:],color='grey',alpha=0.3)
             
    # fig.suptitle('Ht IAC')
        
    # fig,axs = plt.subplots(sbp[0],sbp[1],sharex=True)
    # for p1 in range(sbp[0]):
    #     for p2 in range(sbp[1]):
    #         axs[p1,p2].plot(t,ITD_Ht[p1*sbp[1]+p2,:],color='k')
    #         axs[p1,p2].set_title(ch_picks[p1*sbp[1]+p2])    
    #         for n in range(num_nfs):
    #             axs[p1,p2].plot(t,ITD_Htnf[n][p1*sbp[1]+p2,:],color='grey',alpha=0.3)
            
    # fig.suptitle('Ht ITD')
    
    
    # fig,axs = plt.subplots(sbp2[0],sbp2[1],sharex=True,gridspec_kw=None)
    # for p1 in range(sbp2[0]):
    #     for p2 in range(sbp2[1]):
    #         axs[p1,p2].plot(t,ITD_Ht[p1*sbp2[1]+p2+sbp[0]*sbp[1],:],color='k')
    #         axs[p1,p2].set_title(ch_picks[p1*sbp2[1]+p2+sbp[0]*sbp[1]])   
    #         for n in range(num_nfs):
    #             axs[p1,p2].plot(t,ITD_Htnf[n][p1*sbp2[1]+p2+sbp[0]*sbp[1],:],color='grey',alpha=0.3)
                
    # fig.suptitle('Ht ITD')
        


        
    #%% Save Analysis for subject 
    with open(os.path.join(data_loc, Subject+'_DynBin_SysFunc' + '.pickle'),'wb') as file:     
        pickle.dump([t, IAC_Ht, ITD_Ht, IAC_Htnf, ITD_Htnf, Tot_trials_IAC, Tot_trials_ITD],file)

    del t, IAC_Ht, ITD_Ht, IAC_Htnf, ITD_Htnf, IAC_epochs, ITD_epochs
    
        
        
        
        
    
    
    
    
    
    
    
    
    






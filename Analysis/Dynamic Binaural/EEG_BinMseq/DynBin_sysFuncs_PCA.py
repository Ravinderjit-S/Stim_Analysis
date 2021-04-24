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


Num_noiseFloors = 10


Subjects = ['S001','S132','S203','S204','S205','S206','S207','S208','S211']
Subjects = ['S207']

# tend = 2 #keep Ht until
# tend_ind =  round(tend*2048) -1 

for subj in range(0,len(Subjects)):
    Subject = Subjects[subj]
    print(Subject, '.........................')
    with open(os.path.join(data_loc, Subject+'_DynBin.pickle'),'rb') as f:
        IAC_epochs, ITD_epochs = pickle.load(f)
    
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
    if Subject == 'S208':
        reject_thresh = 800e-6
    else:
        reject_thresh = 400e-6
        
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

    
    #%% Generate Noise Floors
    # Generated by subtracting half of epochs from each other
    
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Commenting out for now - 02/02/20201
    # IAC_nfs = np.zeros([Num_noiseFloors,Mseq.size])
    # ITD_nfs = np.zeros([Num_noiseFloors,Mseq.size])
    # for nn in range(0,Num_noiseFloors):
    #     randInds = np.random.permutation(IAC32.shape[1]) #random inds for epochs
    #     IAC_Noise = (IAC32[:, randInds[0:int(np.round(randInds.size/2))]].sum(axis=1) - \
    #                  IAC32[:, randInds[int(np.round(randInds.size/2)):]].sum(axis=1)) / randInds.size
    #     randInds = np.random.permutation(ITD32.shape[1])
    #     ITD_Noise = (ITD32[:, randInds[0:int(np.round(randInds.size/2))]].sum(axis=1) - \
    #         ITD32[:, randInds[int(np.round(randInds.size/2)):]].sum(axis=1)) / randInds.size
        
    #     IAC_nf = np.correlate(IAC_Noise,Mseq[:,0],mode='full')
    #     IAC_nfs[nn,:] = IAC_nf[Mseq.size-1:]
    #     ITD_nf = np.correlate(ITD_Noise,Mseq[:,0],mode='full')
    #     ITD_nfs[nn,:] = ITD_nf[Mseq.size-1:]
        
    #%% Calculate Ht
    # resp_IAC = IAC_ep.mean(axis=1)
    # resp_ITD = ITD_ep.mean(axis=1)
    
    plt.figure()
    plt.plot(IAC_ep.mean(axis=1))
    plt.plot(ITD_ep.mean(axis=1))
    
    IAC_Ht = mseqXcorr(IAC_ep,Mseq[:,0])
    ITD_Ht = mseqXcorr(ITD_ep,Mseq[:,0])
    
    # IAC_Ht = np.zeros([resp_IAC.shape[0],resp_IAC.shape[1]*2-1])
    # ITD_Ht = np.zeros([resp_ITD.shape[0],resp_ITD.shape[1]*2-1])
    # for ch in ch_picks:
    #     IAC_Ht[ch,:] = np.correlate(resp_IAC[ch,:],Mseq[:,0],mode='full')#[Mseq.size-1:]
    #     ITD_Ht[ch,:] = np.correlate(resp_ITD[ch,:],Mseq[:,0],mode='full')#[Mseq.size-1:]

    # IAC_Ht = IAC_Ht[:,:tend_ind]
    # ITD_Ht = ITD_Ht[:,:tend_ind]
    # t = t[:tend_ind]
    
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
    num_nfs = Num_noiseFloors
        
    if ch_picks.size == 31:
        sbp = [5,3]
        sbp2 = [4,4]
    elif ch_picks.size == 32:
        sbp = [4,4]
        sbp2 = [4,4]
    elif ch_picks.size == 30:
        sbp = [5,3]
        sbp2 = [5,3]

        
    fig,axs = plt.subplots(sbp[0],sbp[1],sharex=True)
    for p1 in range(sbp[0]):
        for p2 in range(sbp[1]):
            axs[p1,p2].plot(t,IAC_Ht[p1*sbp[1]+p2,:],color='k')
            axs[p1,p2].set_title(ch_picks[p1*sbp[1]+p2])    
            for n in range(num_nfs):
                axs[p1,p2].plot(t,IAC_Htnf[n][p1*sbp[1]+p2,:],color='grey',alpha=0.3)
            
    fig.suptitle('Ht IAC')
    
    
    fig,axs = plt.subplots(sbp2[0],sbp2[1],sharex=True,gridspec_kw=None)
    for p1 in range(sbp2[0]):
        for p2 in range(sbp2[1]):
            axs[p1,p2].plot(t,IAC_Ht[p1*sbp2[1]+p2+sbp[0]*sbp[1],:],color='k')
            axs[p1,p2].set_title(ch_picks[p1*sbp2[1]+p2+sbp[0]*sbp[1]])   
            for n in range(num_nfs):
                axs[p1,p2].plot(t,IAC_Htnf[n][p1*sbp2[1]+p2+sbp[0]*sbp[1],:],color='grey',alpha=0.3)
             
    fig.suptitle('Ht IAC')
        
    fig,axs = plt.subplots(sbp[0],sbp[1],sharex=True)
    for p1 in range(sbp[0]):
        for p2 in range(sbp[1]):
            axs[p1,p2].plot(t,ITD_Ht[p1*sbp[1]+p2,:],color='k')
            axs[p1,p2].set_title(ch_picks[p1*sbp[1]+p2])    
            for n in range(num_nfs):
                axs[p1,p2].plot(t,ITD_Htnf[n][p1*sbp[1]+p2,:],color='grey',alpha=0.3)
            
    fig.suptitle('Ht ITD')
    
    
    fig,axs = plt.subplots(sbp2[0],sbp2[1],sharex=True,gridspec_kw=None)
    for p1 in range(sbp2[0]):
        for p2 in range(sbp2[1]):
            axs[p1,p2].plot(t,ITD_Ht[p1*sbp2[1]+p2+sbp[0]*sbp[1],:],color='k')
            axs[p1,p2].set_title(ch_picks[p1*sbp2[1]+p2+sbp[0]*sbp[1]])   
            for n in range(num_nfs):
                axs[p1,p2].plot(t,ITD_Htnf[n][p1*sbp2[1]+p2+sbp[0]*sbp[1],:],color='grey',alpha=0.3)
                
    fig.suptitle('Ht ITD')
        

    
    #%% Calculate Hf
    # nfft = 2**np.ceil(np.log2(IAC_Ht.shape[1]))
    # IAC_Hf = sp.fft(IAC_Ht,n=nfft,axis=1)
    # ITD_Hf = sp.fft(ITD_Ht,n=nfft,axis=1)
    # f = np.arange(0,fs,fs/nfft)
    # Phase_IAC = np.unwrap(np.angle(IAC_Hf))
    # Phase_ITD = np.unwrap(np.angle(ITD_Hf))
    
    #%% Plot Hf
    
    # fig,axs = plt.subplots(sbp[0],sbp[1],sharex=True)
    # for p1 in range(sbp[0]):
    #     for p2 in range(sbp[1]):
    #         axs[p1,p2].plot(f,np.abs(IAC_Hf[p1*sbp[1]+p2,:]),color='k')
    #         axs[p1,p2].set_title(ch_picks[p1*sbp[1]+p2])    
    #         axs[p1,p2].set_xlim(0,30)
    #         for n in range(num_nfs):
    #             nf_f = sp.fft(IAC_Htnf[n][p1*sbp[1]+p2,:],n=nfft)
    #             axs[p1,p2].plot(f,np.abs(nf_f),color='grey',alpha=0.3)
            
    # fig.suptitle('Hf IAC')
    
    
    # fig,axs = plt.subplots(sbp2[0],sbp2[1],sharex=True,gridspec_kw=None)
    # for p1 in range(sbp2[0]):
    #     for p2 in range(sbp2[1]):
    #         axs[p1,p2].plot(f,np.abs(IAC_Hf[p1*sbp2[1]+p2+sbp[0]*sbp[1],:]),color='k')
    #         axs[p1,p2].set_title(ch_picks[p1*sbp2[1]+p2+sbp[0]*sbp[1]])  
    #         axs[p1,p2].set_xlim(0,40)
    #         for n in range(num_nfs):
    #             nf_f = sp.fft(IAC_Htnf[n][p1*sbp2[1]+p2+sbp[0]*sbp[1],:],n=nfft)
    #             axs[p1,p2].plot(f,np.abs(nf_f),color='grey',alpha=0.3)
             
    # fig.suptitle('Hf IAC')
    
    # fig,axs = plt.subplots(sbp[0],sbp[1],sharex=True)
    # for p1 in range(sbp[0]):
    #     for p2 in range(sbp[1]):
    #         axs[p1,p2].plot(f,np.abs(ITD_Hf[p1*sbp[1]+p2,:]),color='k')
    #         axs[p1,p2].set_title(ch_picks[p1*sbp[1]+p2])    
    #         axs[p1,p2].set_xlim(0,40)
    #         # for n in range(m*num_nfs,num_nfs*(m+1)):
    #         #     axs[p1,p2].plot(t,Htnf[n][p1*sbp[1]+p2,:],color='grey',alpha=0.3)
            
    # fig.suptitle('Hf ITD')
    
    
    # fig,axs = plt.subplots(sbp2[0],sbp2[1],sharex=True,gridspec_kw=None)
    # for p1 in range(sbp2[0]):
    #     for p2 in range(sbp2[1]):
    #         axs[p1,p2].plot(f,np.abs(ITD_Hf[p1*sbp2[1]+p2+sbp[0]*sbp[1],:]),color='k')
    #         axs[p1,p2].set_title(ch_picks[p1*sbp2[1]+p2+sbp[0]*sbp[1]])  
    #         axs[p1,p2].set_xlim(0,40)
    #         # for n in range(m*num_nfs,num_nfs*(m+1)):
    #         #     axs[p1,p2].plot(t,Htnf[n][p1*sbp[1]+p2,:],color='grey',alpha=0.3)
             
    # fig.suptitle('Hf ITD')
        
    
    #%% PCA Decomposition
    

    # pca_space_IAC_nf = []
    # pca_f_IAC_nf = []
    # pca_coeffs_IAC_nf =[]
    # pca_expVar_IAC_nf = []
    # n_comp = 2
    
    # pca = PCA(n_components=n_comp)
    # pca_space_IAC = pca.fit_transform(IAC_Ht.T)
    
    # nfft = 2**np.ceil(np.log2(IAC_Ht.shape[1]))
    # pca_f_IAC = sp.fft(pca_space_IAC,n=nfft,axis=0)
    
    # pca_expVar_IAC = pca.explained_variance_ratio_
    # pca_coeff_IAC = pca.components_
    
    # for nf in range(num_nfs):
    #     pca = PCA(n_components=n_comp)
    #     pca_space_IAC_nf.append(pca.fit_transform(IAC_Htnf[nf].T))
    #     pca_f_IAC_nf.append(sp.fft(pca_space_IAC_nf[nf],n=nfft,axis=0))
    #     pca_expVar_IAC_nf.append( pca.explained_variance_ratio_)
    #     pca_coeffs_IAC_nf.append(pca.components_)
        

    # plt.figure()
    # plt.plot(t,pca_space_IAC)
    # for nf in range(num_nfs):
    #     plt.plot(t,pca_space_IAC_nf[nf],color='grey',alpha=0.3)
    # plt.title('PCA IAC')
    
    # plt.figure()
    # plt.plot(f,np.abs(pca_f_IAC))
    # for nf in range(num_nfs):
    #     plt.plot(f,np.abs(pca_f_IAC_nf[nf]),color='grey',alpha=0.3)
    # plt.xlim([0,20])
    # plt.title('PCA IAC')
    
    # vmin = pca_coeff_IAC.mean() - 2 * pca_coeff_IAC.std()
    # vmax = pca_coeff_IAC.mean() + 2 * pca_coeff_IAC.std()
    # plt.figure()
    # mne.viz.plot_topomap(pca_coeff_IAC[0,:], mne.pick_info(IAC_epochs.info, ch_picks),vmin=vmin,vmax=vmax)
    # plt.title('Componenet 1 PCA IAC')
    # plt.figure()
    # mne.viz.plot_topomap(pca_coeff_IAC[1,:], mne.pick_info(IAC_epochs.info, ch_picks),vmin=vmin,vmax=vmax)
    # plt.title('Componenet 2 PCA IAC')
    
    
    # pca_space_ITD_nf = []
    # pca_f_ITD_nf = []
    # pca_coeffs_ITD_nf =[]
    # pca_expVar_ITD_nf = []
    
    # for nf in range(num_nfs):
    #     pca = PCA(n_components=n_comp)
    #     pca_space_ITD_nf.append(pca.fit_transform(ITD_Htnf[nf].T))
    #     pca_f_ITD_nf.append(sp.fft(pca_space_ITD_nf[nf],n=nfft,axis=0))
    #     pca_expVar_ITD_nf.append( pca.explained_variance_ratio_)
    #     pca_coeffs_ITD_nf.append(pca.components_)

    # pca = PCA(n_components=n_comp)
    # pca_space_ITD = pca.fit_transform(ITD_Ht.T)
    
    # nfft = 2**np.ceil(np.log2(ITD_Ht.shape[1]))
    # pca_f_ITD = sp.fft(pca_space_ITD,n=nfft,axis=0)
    
    # pca_expVar_ITD = pca.explained_variance_ratio_
    # pca_coeff_ITD = pca.components_
    
    # plt.figure()
    # plt.plot(t,pca_space_ITD)
    # for nf in range(num_nfs):
    #     plt.plot(t,pca_space_ITD_nf[nf],color='grey',alpha=0.3)
    # plt.title('PCA ITD')
    
    # plt.figure()
    # plt.plot(f,np.abs(pca_f_ITD))
    # for nf in range(num_nfs):
    #     plt.plot(f,np.abs(pca_f_ITD_nf[nf]),color='grey',alpha=0.3)
    # plt.xlim([0,20])
    # plt.title('PCA ITD')
    
    # vmin = pca_coeff_ITD.mean() - 2 * pca_coeff_ITD.std()
    # vmax = pca_coeff_ITD.mean() + 2 * pca_coeff_ITD.std()
    # plt.figure()
    # mne.viz.plot_topomap(pca_coeff_ITD[0,:], mne.pick_info(ITD_epochs.info, ch_picks),vmin=vmin,vmax=vmax)
    # plt.title('Componenet 1 PCA ITD')
    # plt.figure()
    # mne.viz.plot_topomap(pca_coeff_ITD[1,:], mne.pick_info(ITD_epochs.info, ch_picks),vmin=vmin,vmax=vmax)
    # plt.title('Componenet 2 PCA ITD')
    
    #%% Complex PCA
    # C = np.dot(IAC_Hf,IAC_Hf.T) / IAC_Hf.shape[0]
    # U,s,Vh = sp.linalg.svd(IAC_Hf.T)

    
    #%% ICA Decomposition
    # n_comp = 1
    
    # ica_space_ITD_nf = []
    # ica_f_ITD_nf = []
    # ica_coeffs_ITD_nf =[]
    
    # for nf in range(num_nfs):
    #     ica = FastICA(n_components=n_comp)
    #     ica_space_ITD_nf.append(ica.fit_transform(ITD_Htnf[nf].T))
    #     ica_f_ITD_nf.append(sp.fft(ica_space_ITD_nf[nf],n=nfft,axis=0))
    #     ica_coeffs_ITD_nf.append(ica.whitening_)
        
        
    # ica_space_IAC_nf = []
    # ica_f_IAC_nf = []
    # ica_coeffs_IAC_nf =[]
    
    # for nf in range(num_nfs):
    #     ica = FastICA(n_components=n_comp)
    #     ica_space_IAC_nf.append(ica.fit_transform(IAC_Htnf[nf].T))
    #     ica_f_IAC_nf.append(sp.fft(ica_space_IAC_nf[nf],n=nfft,axis=0))
    #     ica_coeffs_IAC_nf.append(ica.whitening_)
        
        
    
    # ica = FastICA(n_components=n_comp)
    # ica_space_IAC = ica.fit_transform(IAC_Ht.T)
    
    # nfft = 2**np.ceil(np.log2(IAC_Ht.shape[1]))
    # ica_f_IAC = sp.fft(ica_space_IAC,n=nfft,axis=0)
    
    # ica_coeff_IAC = ica.whitening_
    
    # plt.figure()
    # plt.plot(t,ica_space_IAC)
    # plt.title('ICA IAC')
    # for nf in range(num_nfs):
    #     plt.plot(t,ica_space_IAC_nf[nf],color='grey',alpha=0.3)
        
    # plt.figure()
    # plt.plot(f,np.abs(ica_f_IAC))
    # for nf in range(num_nfs):
    #     plt.plot(f,np.abs(ica_f_IAC_nf[nf]),color='grey',alpha=0.3)
    # plt.title('ICA IAC')
    # plt.xlim([0,20])
    
    # vmin = ica_coeff_IAC.mean() - 2 * ica_coeff_IAC.std()
    # vmax = ica_coeff_IAC.mean() + 2 * ica_coeff_IAC.std()

    # plt.figure()
    # mne.viz.plot_topomap(ica_coeff_IAC[0,:], mne.pick_info(IAC_epochs.info, ch_picks),vmin=vmin,vmax=vmax)
    # plt.title('ICA IAC')


    # ica = FastICA(n_components=n_comp)
    # ica_space_ITD = ica.fit_transform(ITD_Ht.T)
    
    # nfft = 2**np.ceil(np.log2(ITD_Ht.shape[1]))
    # ica_f_ITD = sp.fft(ica_space_ITD,n=nfft,axis=0)
    
    # ica_coeff_ITD = ica.whitening_
    
    # plt.figure()
    # plt.plot(t,ica_space_ITD)
    # for nf in range(num_nfs):
    #     plt.plot(t,ica_space_ITD_nf[nf],color='grey',alpha=0.3)
    
    # plt.title('ICA ITD')
    
    # plt.figure()
    # plt.xlim([0,20])
    # plt.plot(f,np.abs(ica_f_ITD))
    # for nf in range(num_nfs):
    #     plt.plot(f,np.abs(ica_f_ITD_nf[nf]),color='grey',alpha=0.3)
    # plt.title('ICA ITD')
    
    # vmin = ica_coeff_ITD.mean() - 2 * ica_coeff_ITD.std()
    # vmax = ica_coeff_ITD.mean() + 2 * ica_coeff_ITD.std()
    # plt.figure()
    # mne.viz.plot_topomap(ica_coeff_ITD[0,:], mne.pick_info(IAC_epochs.info, ch_picks),vmin=vmin,vmax=vmax)
    # plt.title('ICA ITD')

        
    #%% Save Analysis for subject
    
    
    # with open(os.path.join(data_loc,'SystemFuncs32', Subject+'_DynBin_SysFunc' + '.pickle'),'wb') as file:     
    #     pickle.dump([t,f,IAC_Ht,ITD_Ht,IAC_Htnf,ITD_Htnf,IAC_Hf,ITD_Hf,
    #                   pca_space_IAC,pca_f_IAC,pca_coeff_IAC,pca_expVar_IAC,
    #                   pca_space_ITD,pca_f_ITD,pca_coeff_ITD,pca_expVar_ITD, 
    #                   pca_space_IAC_nf,pca_f_IAC_nf,pca_coeffs_IAC_nf,
    #                   pca_expVar_IAC_nf, pca_space_ITD_nf,pca_f_ITD_nf,
    #                   pca_coeffs_ITD_nf,pca_expVar_ITD_nf, ica_space_ITD,
    #                   ica_f_ITD,ica_coeff_ITD, ica_space_IAC, ica_f_IAC, 
    #                   ica_coeff_IAC, ica_space_ITD_nf,ica_f_ITD_nf,ica_coeffs_ITD_nf,
    #                   ica_space_IAC_nf, ica_f_IAC_nf,ica_coeffs_IAC_nf],file)

    
    
    
    # with open(os.path.join(data_loc,'SystemFuncs32_2', Subject+'_DynBin_SysFunc' + '.pickle'),'wb') as file:     
    with open(os.path.join(data_loc, Subject+'_DynBin_SysFunc' + '.pickle'),'wb') as file:     
        # pickle.dump([t,IAC_Ht,ITD_Ht,IAC_Htnf,ITD_Htnf,
        #               pca_space_IAC,pca_coeff_IAC,pca_expVar_IAC,
        #               pca_space_ITD,pca_coeff_ITD,pca_expVar_ITD, 
        #               pca_space_IAC_nf,pca_coeffs_IAC_nf,
        #               pca_expVar_IAC_nf, pca_space_ITD_nf,
        #               pca_coeffs_ITD_nf,pca_expVar_ITD_nf, ica_space_ITD,
        #               ica_coeff_ITD, ica_space_IAC, ica_coeff_IAC, 
        #               ica_space_ITD_nf,ica_f_ITD_nf,ica_coeffs_ITD_nf,
        #               ica_space_IAC_nf,ica_coeffs_IAC_nf],file)
        pickle.dump([t, IAC_Ht, ITD_Ht, IAC_Htnf, ITD_Htnf, Tot_trials_IAC, Tot_trials_ITD],file)

    
    
    del t, IAC_Ht, ITD_Ht, IAC_Htnf, ITD_Htnf, IAC_epochs, ITD_epochs
    
        
        
        
        
    
    
    
    
    
    
    
    
    






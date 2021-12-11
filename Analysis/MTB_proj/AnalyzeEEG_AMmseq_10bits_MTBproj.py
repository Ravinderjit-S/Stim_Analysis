#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 09:51:21 2021

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
from mseqHelper import mseqXcorrEpochs_fft
from sklearn.decomposition import PCA


nchans = 34;
refchans = ['EXG1','EXG2']

fig_loc =  '/media/ravinderjit/Data_Drive/Data/Figures/MTBproj/mTRF/'

mseq_loc = '/media/ravinderjit/Data_Drive/Data/EEGdata/TemporalCoding/mseqEEG_150_bits10_4096.mat'
Mseq_dat = sio.loadmat(mseq_loc)
mseq = Mseq_dat['mseqEEG_4096'].astype(float)

data_loc = '/media/ravinderjit/Data_Drive/Data/EEGdata/MTB/mTRF/'
pickle_loc = data_loc + 'Pickles/'

Subjects = ['S078', 'S259','S268', 'S269','S270','S271','S273', 'S274' ,'S277','S279','S281','S282', 'S285','S290']

exclude = ['EXG3','EXG4','EXG5','EXG6','EXG7','EXG8']; #don't need these extra external channels that are saved

num_nfs = 1

for subject in Subjects:
    print('On Subject ................... ' + subject)
    
    #%% Load data and filter
    
    datapath =  os.path.join(data_loc, subject)
    data_eeg,data_evnt = EEGconcatenateFolder(datapath+'/',nchans,refchans,exclude)
    data_eeg.filter(l_freq=1,h_freq=200)
    
    #%% Remove bad channels
    
    if subject == 'S207':
        data_eeg.info['bads'].append('A15') 
        
    if subject == 'S078':
        data_eeg.info['bads'].append('A6')
        
    if subject == 'S281':
        data_eeg.info['bads'].append('A28')
        
    
    
    #%% Blink Removal
    blinks = find_blinks(data_eeg,ch_name = ['A1'],thresh = 100e-6, l_trans_bandwidth = 0.5, l_freq =1.0)
    blink_epochs = mne.Epochs(data_eeg,blinks,998,tmin=-0.25,tmax=0.25,proj=False,
                                  baseline=(-0.25,0),reject=dict(eeg=500e-6))
    Projs = compute_proj_epochs(blink_epochs,n_grad=0,n_mag=0,n_eeg=8)
    
    ocular_projs = Projs[0]
    
    data_eeg.add_proj(ocular_projs)
    data_eeg.plot_projs_topomap()
    plt.savefig(os.path.join(fig_loc,'OcularProjs', subject + '_OcularProjs.png'),format='png')
    
    #data_eeg.plot(events=blinks,show_options=True)
    
    del blinks, blink_epochs, Projs,ocular_projs
    
    #%% Plot data
    fs = data_eeg.info['sfreq']
    reject = dict(eeg=1000e-6)
    epochs = mne.Epochs(data_eeg,data_evnt,1,tmin=-0.3,tmax=np.ceil(mseq.size/fs)+0.4, reject = reject, baseline=(-0.1,0.))     
    epochs.average().plot(picks=[31],titles='10 bit AMmseq')
    plt.savefig(os.path.join(fig_loc,'Evoked', subject + '_OcularProjs.png'),format='png')
    
    #%% Extract part of response when stim is on
    ch_picks = np.arange(32)
    remove_chs = []
    
    if subject == 'S207':
        remove_chs = [14]
        
    elif subject == 'S078':
        remove_chs = [5]
        
    elif subject =='S281':
        reomve_chs = [27]
    
    
    ch_picks = np.delete(ch_picks,remove_chs)
    
    t = epochs.times
    t1 = np.where(t>=0)[0][0]
    t2 = t1 + mseq.size + int(np.round(0.4*fs))
    epdat = epochs.get_data()[:,ch_picks,t1:t2].transpose(1,0,2)
    t = t[t1:t2]
    t = np.concatenate((-t[-int(np.round(0.4*fs)):0:-1],t[:-1]))
    
    info_obj = epochs[0].info
    #del epochs
    #%% Remove epochs with large deflections
    
    Reject_Thresh = 250e-6
    if subject == 'S207':
        Reject_Thresh = 300e-6
    elif subject == 'S228':
        Reject_Thresh = 250e-6
    elif subject == 'S236':
        Reject_Thresh = 300e-6
    elif subject == 'S246':
        Reject_Thresh = 250e-6
    
    epdat = epdat[-1,:,:] #only look at Cz for now
    Reject_Thresh = 100e-6 #if only look at CZ, use this

    epdat = epdat[np.newaxis,:,:]
    
    Peak2Peak = epdat.max(axis=2) - epdat.min(axis=2)
    mask_trials = np.all(Peak2Peak < Reject_Thresh,axis=0)
    print('rejected ' + str(epdat.shape[1] - sum(mask_trials)) + ' trials due to P2P')
    epdat = epdat[:,mask_trials,:]
    print('Total Trials Left: ' + str(epdat.shape[1]))
    Tot_trials = epdat.shape[1]
    
    ch_maxAmp = Peak2Peak.mean(axis=1).argmax()
    plt.figure()
    plt.plot(Peak2Peak.T)
    plt.title(subject + ' ' + str(Tot_trials))
    
    # plt.figure()
    # plt.plot(Peak2Peak[:,mask_trials].T)
    
    # plt.figure()
    # plt.plot(Peak2Peak[np.delete(ch_picks,ch_maxAmp),:].T)
    
    #%% Correlation Analysis
    
    # Htnf = []
    
    Ht = mseqXcorr(epdat,mseq[0,:])
    Ht_epochs, t_epochs = mseqXcorrEpochs_fft(epdat,mseq[0,:],fs)
    
    # for nf in range(num_nfs):
    #     print('On nf:' + str(nf))
    #     resp = epdat.copy()
    #     inv_inds = np.random.permutation(epdat.shape[1])[:round(epdat.shape[1]/2)]
    #     resp[:,inv_inds,:] = -resp[:,inv_inds,:]
    #     Ht_nf = mseqXcorr(resp,mseq[0,:])
    #     Htnf.append(Ht_nf)
    
    #%% Plot mTRF
    
    plt.figure()
    plt.plot(t_epochs,Ht_epochs.mean(axis=1).T,color='k')
    plt.title(subject + ' modTRF Cz')
    plt.xlabel('Time(sec)')
    plt.xlim([-0.1,0.6])
    plt.savefig(os.path.join(fig_loc, subject + '_modTRF_cz.png'),format='png')
    
    #%% Save Data
    
    # with open(os.path.join(pickle_loc,subject+'_AMmseq10bits.pickle'),'wb') as file:
    #     pickle.dump([t, Tot_trials, Ht, Htnf, info_obj, ch_picks],file)
    
    Ht_epochs = Ht_epochs[-1,:,:]
    with open(os.path.join(pickle_loc,subject+'_AMmseq10bits_epochs_cz.pickle'),'wb') as file:
        pickle.dump([Ht_epochs,t_epochs],file)
    
    del data_eeg, data_evnt, epdat, t, Ht, info_obj, Ht_epochs,t_epochs #,Htnf
    
    
    
    
     
    
    
    
    
    
    
    
    
    
    
    

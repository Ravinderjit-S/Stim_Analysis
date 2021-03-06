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
import pickle

import sys
sys.path.append(os.path.abspath('../mseqAnalysis/'))
from mseqHelper import mseqXcorr



nchans = 34;
refchans = ['EXG1','EXG2']

data_eeg = [];
data_evnt = [];
 
#data_loc = '/media/ravinderjit/Storage2/EEGdata'
data_loc = '/media/ravinderjit/Data_Drive/Data/EEGdata/CMR_mseqTarget/CMR_mseq150_10bit'
subject = 'S211'
exclude = ['EXG3','EXG4','EXG5','EXG6','EXG7','EXG8']



Mseq_loc = '/media/ravinderjit/Data_Drive/Data/EEGdata/TemporalCoding/mseqEEG_150_bits10_4096.mat' 
Mseq_dat = sio.loadmat(Mseq_loc)
mseq = Mseq_dat['mseqEEG_4096'].astype(float)

datapath = os.path.join(data_loc,subject)

data_eeg,data_evnt = EEGconcatenateFolder(datapath+'/',nchans,refchans,exclude)
data_eeg.filter(l_freq=1,h_freq=1000)

# if subject == 'S207':
#     data_eeg.info['bads'].append('A15') 
#     data_eeg.info['bads'].append('A17') 

if subject == 'S211':
    data_eeg.info['bads'].append('A25')

    

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
labels = ['Coh 0', 'Coh 1', 'Just mseq']

tmin = -0.3
tmax = 7.3
baseline = (-0.2,0)
reject = dict(eeg=500e-6)
epochs=[]
for m in range(3):
    epochs.append(mne.Epochs(data_eeg,data_evnt,[m+1],tmin=tmin,tmax=tmax,
                              baseline=baseline,reject=reject))
    epochs[m].average().plot(picks=[31],titles=labels[m])
    
    
# epochs_1 = mne.Epochs(data_eeg,data_evnt,[1],tmin=tmin,tmax=tmax,
#                               baseline=baseline)#,reject=reject)
# epochs_2 = mne.Epochs(data_eeg,data_evnt,[2],tmin=tmin,tmax=tmax,
#                               baseline=baseline)#,reject=reject)



#%% Extract part of response when stim is on
ch_picks = np.arange(32)
remove_chs = []
fs = epochs[0].info['sfreq']

if subject =='S211':
    remove_chs = [24] #A25 bad

ch_picks = np.delete(ch_picks,remove_chs)

t = epochs[0].times
t1 = np.where(t>=0)[0][0]
t2 = t1 + mseq.size + int(round(fs*0.4)) #go 300 ms past end of stim
t = t[t1:t2]
t = np.concatenate((-t[-int(np.round(0.4*fs)):0:-1],t[:-1]))
fs = epochs[0].info['sfreq']


epdat = []
for m in range(3):
    epdat.append(epochs[m].get_data()[:,ch_picks,t1:t2].transpose(1,0,2))


#%% Remove epochs with large deflections
# Reject_Thresh = 150e-6


# for m in range(len(epdat)):
#     Peak2Peak = epdat[m].max(axis=2) - epdat[m].min(axis=2)
#     mask_trials = np.all(Peak2Peak <Reject_Thresh,axis=0)
#     print('rejected ' + str(epdat[m].shape[1] - sum(mask_trials)) + ' trials due to P2P')
#     epdat[m] = epdat[m][:,mask_trials,:]
#     print('Total Trials Left: ' + str(epdat[m].shape[1]))
    
# plt.figure()
# plt.plot(Peak2Peak.T)



#%% correlation analysis


Ht = []
Htnf = []
# do cross corr
for m in range(len(epdat)): 
    print('On mseq # ' + str(m+1))
    # resp_m = epdat[m].mean(axis=1)
    Ht_m = mseqXcorr(epdat[m],mseq[0,:])
    Ht.append(Ht_m)


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
elif ch_picks.size ==29:
    sbp = [5,4]
    sbp2 = [3,3]

fig,axs = plt.subplots(sbp[0],sbp[1],sharex=True,gridspec_kw=None)
for p1 in range(sbp[0]):
    for p2 in range(sbp[1]):
        axs[p1,p2].plot(t,Ht[0][p1*sbp[1]+p2,:],color='b')
        axs[p1,p2].plot(t,Ht[1][p1*sbp[1]+p2,:],color='r')
        axs[p1,p2].plot(t,Ht[2][p1*sbp[1]+p2,:],color='k')
        axs[p1,p2].set_title(ch_picks[p1*sbp[1]+p2])    

fig.suptitle('Ht ' + labels[m])


fig,axs = plt.subplots(sbp2[0],sbp2[1],sharex=True,gridspec_kw=None)
for p1 in range(sbp2[0]):
    for p2 in range(sbp2[1]):
        axs[p1,p2].plot(t,Ht[0][p1*sbp2[1]+p2+sbp[0]*sbp[1],:],color='b')
        axs[p1,p2].plot(t,Ht[1][p1*sbp2[1]+p2+sbp[0]*sbp[1],:],color='r')
        axs[p1,p2].plot(t,Ht[2][p1*sbp2[1]+p2+sbp[0]*sbp[1],:],color='k')
        axs[p1,p2].set_title(ch_picks[p1*sbp2[1]+p2+sbp[0]*sbp[1]])   

fig.suptitle('Ht ' + labels[m])    
    



#%% Plot Hf
# Hf = []
# Phase = []
# #compute HF
# nfft = 2**np.log2(Ht[m].shape[1])
# f = np.arange(0,fs,fs/nfft)
# for m in range(len(Ht)):
#     Hf.append(sp.fft(Ht[m],n=nfft,axis=1))
#     Phase.append(np.unwrap(np.angle(Hf[m])))
    


# fig,axs = plt.subplots(sbp[0],sbp[1],sharex=True,gridspec_kw=None)
# for p1 in range(sbp[0]):
#     for p2 in range(sbp[1]):
#         axs[p1,p2].plot(f,np.abs(Hf[0][p1*sbp[1]+p2,:]),color='b')
#         axs[p1,p2].plot(f,np.abs(Hf[1][p1*sbp[1]+p2,:]),color='r')
#         axs[p1,p2].plot(f,np.abs(Hf[2][p1*sbp[1]+p2,:]),color='k')
#         axs[p1,p2].set_title(ch_picks[p1*sbp[1]+p2])  
#         axs[p1,p2].set_xlim([0,150])
# fig.suptitle('Hf ' + labels[m])


# fig,axs = plt.subplots(sbp2[0],sbp2[1],sharex=True,gridspec_kw=None)
# for p1 in range(sbp2[0]):
#     for p2 in range(sbp2[1]):
#         axs[p1,p2].plot(f,np.abs(Hf[0][p1*sbp2[1]+p2+sbp[0]*sbp[1],:]),color='b')
#         axs[p1,p2].plot(f,np.abs(Hf[1][p1*sbp2[1]+p2+sbp[0]*sbp[1],:]),color='r')
#         axs[p1,p2].plot(f,np.abs(Hf[2][p1*sbp2[1]+p2+sbp[0]*sbp[1],:]),color='k')
#         axs[p1,p2].set_title(ch_picks[p1*sbp2[1]+p2+sbp[0]*sbp[1]])  
#         axs[p1,p2].set_xlim([0,150])
# fig.suptitle('Hf ' + labels[m])  




#%% PCA decomposition of Ht
# pca_sp = []
# pca_fft = []
# pca_phase = []
# pca_coeff = []
# pca_expVar = []

# pca_sp_nf = []
# pca_fft_nf = []
# pca_phase_nf = []
# pca_coeff_nf = []
# pca_expVar_nf = []

# n_comp = 2

# for m in range(len(Ht)):
#     pca = PCA(n_components=n_comp)
#     pca.fit(Ht[m])
#     pca_space = pca.fit_transform(Ht[m].T)
    
#     pca_sp.append(pca_space)
#     pca_coeff.append(pca.components_)
#     pca_expVar.append(pca.explained_variance_ratio_)
    
# # for n in range(len(Htnf)):
# #     pca = PCA(n_components=n_comp)
# #     pca.fit(Htnf[n])
# #     pca_space = pca.fit_transform(Htnf[n].T)
    
# #     nfft=2**np.log2(Htnf[n].shape[1])
# #     pca_f = sp.fft(pca_space,n=nfft,axis=0)
    
# #     pca_sp_nf.append(pca_space)
# #     pca_fft_nf.append(pca_f)
# #     pca_coeff_nf.append(pca.components_)
# #     pca_expVar_nf.append(pca.explained_variance_ratio_)
    
# for m in range(len(pca_sp)):
#     fig,axs = plt.subplots(2,1)
#     axs[0].plot(t,pca_sp[m])
#     # for n in range(m*num_nfs,num_nfs*(m+1)):
#     #     axs[0].plot(tdat[m],pca_sp_nf[n],color='grey',alpha=0.3)
    

    
#     axs[1].plot(ch_picks,pca_coeff[m].T)
#     axs[1].set_xlabel('channel')
#     # for n in range(m*num_nfs,num_nfs*(m+1)):
#     #     axs[3].plot(ch_picks,pca_coeff_nf[n].T,color='grey',alpha=0.1)
#     fig.suptitle('PCA ' + labels[m])    
    
# p_ind = 2
# vmin = pca_coeff[p_ind].mean() - 2 * pca_coeff[p_ind].std()
# vmax = pca_coeff[p_ind].mean() + 2 * pca_coeff[p_ind].std()
# plt.figure()
# mne.viz.plot_topomap(pca_coeff[p_ind][0,:], mne.pick_info(epochs[0].info, ch_picks),vmin=vmin,vmax=vmax)
# plt.figure()
# mne.viz.plot_topomap(pca_coeff[p_ind][1,:], mne.pick_info(epochs[0].info, ch_picks),vmin=vmin,vmax=vmax)
# # plt.figure()
# # mne.viz.plot_topomap(pca_coeff[2][2,:], mne.pick_info(epochs[3].info, ch_picks),vmin=vmin,vmax=vmax)


#%% PCA decomposition on t_splits
t_cuts = [.015,.040,.125,.500]

pca_sp_cuts_cond = []
pca_expVar_cuts_cond = []
pca_coeff_cuts_cond = []
pca = PCA(n_components=1)
t__ = [list() for i in range(len(t_cuts))]

for m in range(len(Ht)): #each condition
    pca_sp_cuts = [list() for i in range(len(t_cuts))]
    pca_expVar_cuts = [list() for i in range(len(t_cuts))]
    pca_coeff_cuts = [list() for i in range(len(t_cuts))]
    for t_c in range(len(t_cuts)):
        if t_c ==0:
            t_1 = np.where(t>=0)[0][0]
        else:
            t_1 = np.where(t>=t_cuts[t_c-1])[0][0]
        t_2 = np.where(t>=t_cuts[t_c])[0][0]
        
        pca_sp_cuts[t_c] = pca.fit_transform(Ht[m][:,t_1:t_2].T)
        pca_expVar_cuts[t_c] = pca.explained_variance_ratio_
        pca_coeff_cuts[t_c] = pca.components_
        
        if pca_coeff_cuts[t_c][0,-1] < 0:
            pca_coeff_cuts[t_c] = -pca_coeff_cuts[t_c]
            pca_sp_cuts[t_c] = -pca_sp_cuts[t_c]
            
        t__[t_c] = t[t_1:t_2]
    
    pca_sp_cuts_cond.append(pca_sp_cuts)
    pca_expVar_cuts_cond.append(pca_expVar_cuts)
    pca_coeff_cuts_cond.append(pca_coeff_cuts)



for t_c in range(len(t_cuts)):
    plt.figure()
    for m in range(len(Ht)):
        plt.plot(t__[t_c],pca_sp_cuts_cond[m][t_c])
    
    
vmin = pca_coeff_cuts_cond[0][0].mean() - 2 * pca_coeff_cuts_cond[0][0].std()
vmax = pca_coeff_cuts_cond[0][0].mean() + 2 * pca_coeff_cuts_cond[0][0].std()
for m in range(len(Ht)):
    plt.figure()
    for t_c in range(len(t_cuts)):
        plt.subplot(1,len(t_cuts),t_c+1)
        mne.viz.plot_topomap(pca_coeff_cuts_cond[m][t_c][0,:], mne.pick_info(epochs[0].info, ch_picks),vmin=vmin,vmax=vmax)

        

# p_ind = 1
# vmin = pca_coeff_s1[p_ind].mean() - 2 * pca_coeff_s1[p_ind].std()
# vmax = pca_coeff_s1[p_ind].mean() + 2 * pca_coeff_s1[p_ind].std()
# plt.figure()
# mne.viz.plot_topomap(pca_coeff_s1[p_ind][0,:], mne.pick_info(epochs[0].info, ch_picks),vmin=vmin,vmax=vmax)
# plt.figure()
# mne.viz.plot_topomap(pca_coeff_s2[p_ind][0,:], mne.pick_info(epochs[0].info, ch_picks),vmin=vmin,vmax=vmax)

#%% PCA decomposition with template

pca_template_loc = '/media/ravinderjit/Data_Drive/Data/EEGdata/TemporalCoding/AMmseq_bits4/Pickles_full/PCA_tsplit_temp.pickle'
with open(pca_template_loc,'rb') as file:
    [t_cuts,A_t_,pca_coeff_cuts,pca_sp_cuts,pca_expVar_cuts] = pickle.load(file)

pca_sp_cuts_temp_cond = []
pca = PCA(n_components=1)
t__ = [list() for i in range(len(t_cuts))]

 
for m in range(len(Ht)): #each condition
    pca_sp_cuts = [list() for i in range(len(t_cuts))]

    
    for t_c in range(len(t_cuts)):
        if t_c ==0:
            t_1 = np.where(t>=0)[0][0]
        else:
            t_1 = np.where(t>=t_cuts[t_c-1])[0][0]
        t_2 = np.where(t>=t_cuts[t_c])[0][0]
        
        Ht_segment = Ht[m][:,t_1:t_2].T
        Ht_segment = Ht_segment - Ht_segment.mean(axis=0)[np.newaxis,:] #demean
        pca_sp_cuts[t_c] = np.matmul(Ht_segment,pca_coeff_cuts[t_c][0,ch_picks].T)
     
        t__[t_c] = t[t_1:t_2]
    
    pca_sp_cuts_temp_cond.append(pca_sp_cuts)



for t_c in range(len(t_cuts)):
    plt.figure()
    for m in range(len(Ht)):
        plt.plot(t__[t_c],pca_sp_cuts_temp_cond[m][t_c])








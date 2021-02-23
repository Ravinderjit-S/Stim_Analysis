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
  
mseq_locs = ['mseqEEG_150_bits7_4096.mat', 'mseqEEG_150_bits8_4096.mat', 
             'mseqEEG_150_bits9_4096.mat', 'mseqEEG_150_bits10_4096.mat']
mseq = []
for m in mseq_locs:
    file_loc = '/media/ravinderjit/Data_Drive/Data/EEGdata/TemporalCoding/' + m
    Mseq_dat = sio.loadmat(file_loc)
    mseq.append( Mseq_dat['mseqEEG_4096'].astype(float) )



#data_loc = '/media/ravinderjit/Storage2/EEGdata/'
data_loc = '/media/ravinderjit/Data_Drive/Data/EEGdata/TemporalCoding/AMmseq_bits4/'
subject = 'S211'
exclude = ['EXG3','EXG4','EXG5','EXG6','EXG7','EXG8']; #don't need these extra external channels that are saved

datapath =  os.path.join(data_loc, subject)
# datapath = '/media/ravinderjit/Data_Drive/Data/EEGdata/EFR'
data_eeg,data_evnt = EEGconcatenateFolder(datapath+'/',nchans,refchans,exclude)
data_eeg.filter(l_freq=1,h_freq=200)

if subject == 'S211':
    data_eeg.info['bads'].append('A10') #Channel A10 bad in S211

#%% Blink Removal
blinks = find_blinks(data_eeg,ch_name = ['A1'],thresh = 100e-6, l_trans_bandwidth = 0.5, l_freq =1.0)
blink_epochs = mne.Epochs(data_eeg,blinks,998,tmin=-0.25,tmax=0.25,proj=False,
                              baseline=(-0.25,0),reject=dict(eeg=500e-6))
Projs = compute_proj_epochs(blink_epochs,n_grad=0,n_mag=0,n_eeg=8,verbose='DEBUG')
blink_proj = Projs[0]

data_eeg.add_proj(blink_proj)
data_eeg.plot_projs_topomap()
data_eeg.plot(events=blinks,show_options=True)

#%% Plot data

labels = ['7bits', '8bits','9bits','10bits']
reject = None
epochs = []
for j in range(len(mseq)):
    epochs.append(mne.Epochs(data_eeg, data_evnt, [j+1], tmin=-0.3, 
                             tmax=np.ceil(mseq[j].size/4096),reject=reject, baseline=(-0.2, 0.)) )
    epochs[j].average().plot(picks=[31],titles = labels[j])


#del data_eeg, data_evnt, evkd_data_1, evkd_data_2 #, 

#%% Extract part of response when stim is on
ch_picks = np.arange(32)
remove_chs = [9,10] # channel A10 bad in S211
ch_picks = np.delete(ch_picks,remove_chs)

epdat = []
tdat = []
for m in range(len(mseq)):
    t = epochs[m].times
    t1 = np.where(t>=0)[0][0]
    t2 = t1 + mseq[m].size
    epdat.append(epochs[m].get_data()[:,ch_picks,t1:t2].transpose(1,0,2))
    tdat.append(t[t1:t2])
    
fs = epochs[0].info['sfreq']

#%% Remove epochs with large deflections

for m in range(len(mseq)):
    Peak2Peak = epdat[m].max(axis=2) - epdat[m].min(axis=2)
    mask_trials = np.all(Peak2Peak <100e-6,axis=0)
    print('rejected ' + str(epdat[m].shape[1] - sum(mask_trials)) + ' trials due to P2P')
    epdat[m] = epdat[m][:,mask_trials,:]
    


# params = dict()
# params['Fs'] = fs
# params['tapers'] = [2,2*2-1]
# params['fpass'] = [0, 200]
# params['itc'] = 0

# TW = 12
# Fres = (1/t[-1]) * TW * 2

#PLV_AM, Coh_AM, f, Phase_AM = PLV_Coh(mseq2,AMch_2,TW,fs)
# PLV_FM, Coh_FM, f, Phase_FM = PLV_Coh(mseq,FMdata,TW,fs)


#%% Correlation Analysis

tend = 0.5 #time of Ht to keep
tend_ind = round(tend*fs) - 1

num_nfs = 10


Ht = []
Htnf = []
# do cross corr
for m in range(len(epdat)): 
    print('On mseq # ' + str(m+1))
    resp_m = epdat[m].mean(axis=1)
    Ht_m = np.zeros(resp_m.shape)
    for ch in range(resp_m.shape[0]):
        Ht_m[ch,:] = np.correlate(resp_m[ch,:],mseq[m][0,:],mode='full')[mseq[m].size-1:]
    Ht.append(Ht_m)
    for nf in range(num_nfs):
        resp = epdat[m]
        inv_inds = np.random.permutation(epdat[m].shape[1])[:round(epdat[m].shape[1]/2)]
        resp[:,inv_inds,:] = -resp[:,inv_inds,:]
        resp_nf = resp.mean(axis=1)
        Ht_nf = np.zeros(resp_nf.shape)
        for ch in range(resp_nf.shape[0]):
            Ht_nf[ch,:] = np.correlate(resp_nf[ch,:],mseq[m][0,:],mode='full')[mseq[m].size-1:]
        Htnf.append(Ht_nf[:,:tend_ind])
    
#only keep Ht up to tend 
for h in range(len(Ht)):
    Ht[h] = Ht[h][:,:tend_ind]
    tdat[h] = tdat[h][:tend_ind]



# Plot Ht
    
if ch_picks.size == 31:
    sbp = [5,3]
    sbp2 = [4,4]
elif ch_picks.size == 32:
    sbp = [4,4]
    sbp2 = [4,4]
elif ch_picks.size == 30:
    sbp = [5,3]
    sbp2 = [5,3]
    


for m in range(len(Ht)):
    Ht_1 = Ht[m]
    t = tdat[m]
    fig,axs = plt.subplots(sbp[0],sbp[1],sharex=True,gridspec_kw=None)
    for p1 in range(sbp[0]):
        for p2 in range(sbp[1]):
            axs[p1,p2].plot(t,Ht_1[p1*sbp[1]+p2,:],color='k')
            axs[p1,p2].set_title(ch_picks[p1*sbp[1]+p2])    
            for n in range(m*num_nfs,num_nfs*(m+1)):
                axs[p1,p2].plot(t,Htnf[n][p1*sbp[1]+p2,:],color='grey',alpha=0.3)
            
    fig.suptitle('Ht ' + labels[m])
    
    
    fig,axs = plt.subplots(sbp2[0],sbp2[1],sharex=True,gridspec_kw=None)
    for p1 in range(sbp2[0]):
        for p2 in range(sbp2[1]):
            axs[p1,p2].plot(t,Ht_1[p1*sbp2[1]+p2+sbp[0]*sbp[1],:],color='k')
            axs[p1,p2].set_title(ch_picks[p1*sbp2[1]+p2+sbp[0]*sbp[1]])   
            for n in range(m*num_nfs,num_nfs*(m+1)):
                axs[p1,p2].plot(t,Htnf[n][p1*sbp[1]+p2,:],color='grey',alpha=0.3)
            
    fig.suptitle('Ht ' + labels[m])    
    
Hf = []
Phase = []
fdat = []
#compute HF
for m in range(len(Ht)):
    nfft = 2**np.log2(Ht[m].shape[1])
    Hf.append(sp.fft(Ht[m],n=nfft,axis=1))
    fdat.append(np.arange(0,fs,fs/nfft))
    Phase.append(np.unwrap(np.angle(Hf[m])))

    
# Plot Hf
# sbp = [5,3]
# sbp2 = [5,3]
for m in range(len(Hf)):
    Hf_1 = Hf[m]
    f = fdat[m]
    fig,axs = plt.subplots(sbp[0],sbp[1],sharex=True,gridspec_kw=None)
    for p1 in range(sbp[0]):
        for p2 in range(sbp[1]):
            axs[p1,p2].plot(f,np.abs(Hf_1[p1*sbp[1]+p2,:]))
            axs[p1,p2].set_title(ch_picks[p1*sbp[1]+p2])  
            axs[p1,p2].set_xlim([0,150])
    fig.suptitle('Hf ' + labels[m])
    
    
    fig,axs = plt.subplots(sbp2[0],sbp2[1],sharex=True,gridspec_kw=None)
    for p1 in range(sbp2[0]):
        for p2 in range(sbp2[1]):
            axs[p1,p2].plot(f,np.abs(Hf_1[p1*sbp2[1]+p2+sbp[0]*sbp[1],:]))
            axs[p1,p2].set_title(ch_picks[p1*sbp2[1]+p2+sbp[0]*sbp[1]])  
            axs[p1,p2].set_xlim([0,150])
    fig.suptitle('Hf ' + labels[m])  



#PCA decomposition of Ht
pca_sp = []
pca_fft = []
pca_phase = []
pca_coeff = []
pca_expVar = []

pca_sp_nf = []
pca_fft_nf = []
pca_phase_nf = []
pca_coeff_nf = []
pca_expVar_nf = []

n_comp = 3

for m in range(len(Ht)):
    pca = PCA(n_components=n_comp)
    pca.fit(Ht[m])
    pca_space = pca.fit_transform(Ht[m].T)
    
    nfft = 2**np.log2(Ht[m].shape[1])
    pca_f = sp.fft(pca_space,n=nfft,axis=0)
    
    pca_sp.append(pca_space)
    pca_fft.append(pca_f)
    pca_phase.append(np.unwrap(np.angle(pca_f),axis=0))
    pca_coeff.append(pca.components_)
    pca_expVar.append(pca.explained_variance_ratio_)
    
for n in range(len(Htnf)):
    pca = PCA(n_components=n_comp)
    pca.fit(Htnf[n])
    pca_space = pca.fit_transform(Htnf[n].T)
    
    nfft=2**np.log2(Htnf[n].shape[1])
    pca_f = sp.fft(pca_space,n=nfft,axis=0)
    
    pca_sp_nf.append(pca_space)
    pca_fft_nf.append(pca_f)
    pca_coeff_nf.append(pca.components_)
    pca_expVar_nf.append(pca.explained_variance_ratio_)
    
for m in range(len(pca_sp)):
    fig,axs = plt.subplots(4,1)
    axs[0].plot(tdat[m],pca_sp[m])
    for n in range(m*num_nfs,num_nfs*(m+1)):
        axs[0].plot(tdat[m],pca_sp_nf[n],color='grey',alpha=0.3)
    
    axs[1].plot(fdat[m],np.abs(pca_fft[m]))
    axs[1].set_xlim([0,100])
    for n in range(m*num_nfs,num_nfs*(m+1)):
        axs[1].plot(fdat[m],pca_fft_nf[n],color='grey',alpha=0.3)
    
    axs[2].plot(fdat[m],pca_phase[m])
    axs[2].set_xlim([0,100])
    
    axs[3].plot(ch_picks,pca_coeff[m].T)
    axs[3].set_xlabel('channel')
    for n in range(m*num_nfs,num_nfs*(m+1)):
        axs[3].plot(ch_picks,pca_coeff_nf[n].T,color='grey',alpha=0.1)
    fig.suptitle('PCA ' + labels[m])    
    
    
vmin = pca_coeff[2].mean() - 2 * pca_coeff[2].std()
vmax = pca_coeff[2].mean() + 2 * pca_coeff[2].std()
plt.figure()
mne.viz.plot_topomap(pca_coeff[2][0,:], mne.pick_info(epochs[3].info, ch_picks),vmin=vmin,vmax=vmax)
plt.figure()
mne.viz.plot_topomap(pca_coeff[2][1,:], mne.pick_info(epochs[3].info, ch_picks),vmin=vmin,vmax=vmax)
plt.figure()
mne.viz.plot_topomap(pca_coeff[2][2,:], mne.pick_info(epochs[3].info, ch_picks),vmin=vmin,vmax=vmax)



    

#ICA decomposition of Ht
ica_sp = []
ica_fft = []
ica_phase = []
ica_coeff = []
#ica_expVar = []

ica_sp_nf = []
ica_fft_nf = []
ica_phase_nf = []
ica_coeff_nf = []
#ica_expVar_nf = []

for m in range(len(Ht)):
    ica = FastICA(n_components=n_comp)
    ica.fit(Ht[m])
    ica_space = ica.fit_transform(Ht[m].T)
    
    nfft = 2**np.log2(Ht[m].shape[1])
    ica_f = sp.fft(ica_space,n=nfft,axis=0)
    
    ica_sp.append(ica_space)
    ica_fft.append(ica_f)
    ica_phase.append(np.unwrap(np.angle(ica_f),axis=0))
    ica_coeff.append(ica.components_)
    #ica_expVar.ap pend(ica.explained_variance_ratio_)
    
for n in range(len(Htnf)):
    
    ica = FastICA(n_components=n_comp)
    ica.fit(Ht[m])
    ica_space = ica.fit_transform(Htnf[m].T)
    
    nfft = 2**np.log2(Htnf[m].shape[1])
    ica_f = sp.fft(ica_space,n=nfft,axis=0)
    
    ica_sp_nf.append(ica_space)
    ica_fft_nf.append(ica_f)
    ica_phase_nf.append(np.unwrap(np.angle(ica_f),axis=0))
    ica_coeff_nf.append(ica.components_)
    #ica_expVar_nf.append(ica.explained_variance_ratio_)
    

    
    
for m in range(len(ica_sp)):
    fig,axs = plt.subplots(4,1)
    axs[0].plot(tdat[m],ica_sp[m])
    for n in range(m*num_nfs,num_nfs*(m+1)):
        axs[0].plot(tdat[m],ica_sp_nf[n],color='grey',alpha=0.3)
    
    axs[1].plot(fdat[m],np.abs(ica_fft[m]))
    axs[1].set_xlim([0,100])
    for n in range(m*num_nfs,num_nfs*(m+1)):
        axs[1].plot(fdat[m],ica_fft_nf[n],color='grey',alpha=0.3)
    
    axs[2].plot(fdat[m],ica_phase[m])
    axs[2].set_xlim([0,100])
    
    axs[3].plot(ch_picks,ica_coeff[m].T)
    axs[3].set_xlabel('channel')
    # for n in range(m*num_nfs,num_nfs*(m+1)):
    #     axs[3].plot(ch_picks,ica_coeff_nf[n].T,color='grey',alpha=0.1)
    fig.suptitle('ICA ' + labels[m])    
    
    
vmin = ica_coeff[2].mean() - 2 * ica_coeff[2].std()
vmax = ica_coeff[2].mean() + 2 * ica_coeff[2].std()
plt.figure()
mne.viz.plot_topomap(ica_coeff[2][0,:], mne.pick_info(epochs[3].info, ch_picks),vmin=vmin,vmax=vmax)
plt.figure()
mne.viz.plot_topomap(ica_coeff[2][1,:], mne.pick_info(epochs[3].info, ch_picks),vmin=vmin,vmax=vmax)
plt.figure()
mne.viz.plot_topomap(ica_coeff[2][2,:], mne.pick_info(epochs[3].info, ch_picks),vmin=vmin,vmax=vmax)




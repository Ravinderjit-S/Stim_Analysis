#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 13 17:41:55 2021

@author: ravinderjit
"""
import os
import pickle
import numpy as np
import scipy as sp
import mne
import matplotlib.pyplot as plt
from scipy.signal import freqz
from scipy.signal import hilbert
from scipy import signal
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from scipy.stats import pearsonr
import scipy.io as sio

sent_data = '/media/ravinderjit/Data_Drive/Data/EEGdata/Sentences/'
data_loc = '/media/ravinderjit/Data_Drive/Data/EEGdata/TemporalCoding/AMmseq_bits4/Pickles/'
sentEnv_loc = '/media/ravinderjit/Data_Drive/Stim_Analysis/Stimuli/Sentences/sentEnv.mat'

sentEnv = sio.loadmat(sentEnv_loc)['envs_4096'][0]
subject = 'S211'

with open(os.path.join(sent_data,subject+'_sentEnv.pickle'),'rb') as file:
    [sent1resp, sent2resp] = pickle.load(file)
    
with open(os.path.join(data_loc,subject+'_AMmseqbits4.pickle'),'rb') as file:
    # [tdat, Tot_trials, Ht, Htnf, pca_sp, pca_coeff, pca_expVar, 
    #  pca_sp_nf, pca_coeff_nf,pca_expVar_nf,ica_sp,
    #  ica_coeff,ica_sp_nf,ica_coeff_nf, info_obj, ch_picks] = pickle.load(file)
    [tdat, Tot_trials, Ht, Htnf, info_obj, ch_picks] = pickle.load(file)


t_ACR = tdat[3]
ACR_tlen = 0.3
t1 = np.where(t_ACR>=0)[0][0]
t2 = np.where(t_ACR>=ACR_tlen)[0][0]
t_ACR=t_ACR[t1:t2]
ACR = Ht[3][:,t1:t2]
fs = 4096

#%% PLot time domain Ht


sbp = [4,4]
sbp2 = [4,4]

    

fig,axs = plt.subplots(sbp[0],sbp[1],sharex=True,gridspec_kw=None)
for p1 in range(sbp[0]):
    for p2 in range(sbp[1]):
        cur_ch = p1*sbp[1]+p2
        if np.any(cur_ch==ch_picks):
            ch_ind = np.where(cur_ch==ch_picks)[0][0]
            axs[p1,p2].plot(t_ACR,ACR[ch_ind,:],color='k')
            axs[p1,p2].set_title(ch_picks[ch_ind])    

fig.suptitle('ACR')


fig,axs = plt.subplots(sbp2[0],sbp2[1],sharex=True,gridspec_kw=None)
for p1 in range(sbp2[0]):
    for p2 in range(sbp2[1]):
        cur_ch = p1*sbp2[1]+p2+sbp[0]*sbp[1]
        if np.any(cur_ch==ch_picks):
            ch_ind = np.where(cur_ch==ch_picks)[0][0]
            axs[p1,p2].plot(t_ACR,ACR[ch_ind,:],color='k')
            axs[p1,p2].set_title(ch_picks[ch_ind])   

fig.suptitle('ACR')    

#%% Do Prediction with ACR

y_s1 = np.zeros([ACR.shape[0],ACR.shape[1]+sentEnv[0].size-1])
y_s2 = np.zeros([ACR.shape[0],ACR.shape[1]+sentEnv[1].size-1])
for ch in range(ACR.shape[0]):
    h = ACR[ch,:]
    y_s1[ch,:] = np.convolve(sentEnv[0][:,0] - sentEnv[0][:,0].mean(),h,mode='full')
    y_s2[ch,:] = np.convolve(sentEnv[1][:,0] - sentEnv[1][:,0].mean(),h,mode='full')
    

#%% Extract Real resp to sent
t_sr1 = sent1resp.times
sr1 = sent1resp.data[:32,:]
t1 = np.where(t_sr1>=0)[0][0]
t2 = np.where(t_sr1>= (sentEnv[0].size/fs + ACR_tlen))[0][0] - 1
t_sr1 = t_sr1[t1:t2]
sr1 = sr1[:,t1:t2]


#%% Plot Prediction

#t_env1 = np.arange(0,sentEnv[0].size/fs,1/fs)   
#t_conv1 = np.concatenate([-t_ACR[-1:0:-1],t_env1])
t_conv1 = np.arange(0,y_s1.shape[1]/fs,1/fs)

    
fig,axs = plt.subplots(sbp[0],sbp[1],sharex=True,gridspec_kw=None)
for p1 in range(sbp[0]):
    for p2 in range(sbp[1]):
        cur_ch = p1*sbp[1]+p2
        if np.any(cur_ch==ch_picks):
            ch_ind = np.where(cur_ch==ch_picks)[0][0]
            axs[p1,p2].plot(t_sr1,sr1[cur_ch,:]/np.abs(sr1[cur_ch,:]).max(),color='k')
            axs[p1,p2].plot(t_conv1,y_s1[ch_ind,:]/np.abs(y_s1[ch_ind,:]).max(),color='r')
            axs[p1,p2].set_title(ch_picks[ch_ind])    

fig.suptitle('Y_predict')


fig,axs = plt.subplots(sbp2[0],sbp2[1],sharex=True,gridspec_kw=None)
for p1 in range(sbp2[0]):
    for p2 in range(sbp2[1]):
        cur_ch = p1*sbp2[1]+p2+sbp[0]*sbp[1]
        if np.any(cur_ch==ch_picks):
            ch_ind = np.where(cur_ch==ch_picks)[0][0]
            axs[p1,p2].plot(t_sr1,sr1[cur_ch,:]/np.abs(sr1[cur_ch,:]).max(),color='k')
            axs[p1,p2].plot(t_conv1,y_s1[ch_ind,:] / np.abs(y_s1[ch_ind,:]).max(),color='r')
            axs[p1,p2].set_title(ch_picks[ch_ind])   

fig.suptitle('Y_predict')    

#%% Plot hilbert of prediction and response

lpf_cut = 15
n_filt = int(np.round(4/lpf_cut * fs))
h_lpf = signal.firwin(n_filt,75,fs=fs)

fig,axs = plt.subplots(sbp[0],sbp[1],sharex=True,gridspec_kw=None)
for p1 in range(sbp[0]):
    for p2 in range(sbp[1]):
        cur_ch = p1*sbp[1]+p2
        if np.any(cur_ch==ch_picks):
            ch_ind = np.where(cur_ch==ch_picks)[0][0]
            
            hilbert_sig = hilbert(sr1[cur_ch,:]/np.abs(sr1[cur_ch,:]).max())
            env_sig = np.abs(hilbert_sig)
            env_sig = signal.filtfilt(h_lpf,1,env_sig)
            
            hilbert_pred = hilbert(y_s1[ch_ind,:]/np.abs(y_s1[ch_ind,:]).max())
            env_pred = np.abs(hilbert_pred)
            env_pred = signal.filtfilt(h_lpf,1,env_pred)
            
            axs[p1,p2].plot(t_sr1,env_sig,color='k')
            axs[p1,p2].plot(t_conv1,env_pred,color='r')
            axs[p1,p2].set_title(ch_picks[ch_ind])    

fig.suptitle('Y_predict')


fig,axs = plt.subplots(sbp2[0],sbp2[1],sharex=True,gridspec_kw=None)
for p1 in range(sbp2[0]):
    for p2 in range(sbp2[1]):
        cur_ch = p1*sbp2[1]+p2+sbp[0]*sbp[1]
        if np.any(cur_ch==ch_picks):
            ch_ind = np.where(cur_ch==ch_picks)[0][0]
            
            hilbert_sig = hilbert(sr1[cur_ch,:]/np.abs(sr1[cur_ch,:]).max())
            env_sig = np.abs(hilbert_sig)
            
            hilbert_pred = hilbert(y_s1[ch_ind,:]/np.abs(y_s1[ch_ind,:]).max())
            env_pred = np.abs(hilbert_pred)
            
            axs[p1,p2].plot(t_sr1,env_sig,color='k')
            axs[p1,p2].plot(t_conv1,env_pred,color='r')
            axs[p1,p2].set_title(ch_picks[ch_ind])   

fig.suptitle('Y_predict')    



#%% Plot coherence spectrum of hilbert for prediction and response



fig,axs = plt.subplots(sbp[0],sbp[1],sharex=True,gridspec_kw=None)
for p1 in range(sbp[0]):
    for p2 in range(sbp[1]):
        cur_ch = p1*sbp[1]+p2
        if np.any(cur_ch==ch_picks):
            ch_ind = np.where(cur_ch==ch_picks)[0][0]
            
            hilbert_sig = hilbert(sr1[cur_ch,:]/np.abs(sr1[cur_ch,:]).max())
            env_sig = np.abs(hilbert_sig)
            env_sig = signal.filtfilt(h_lpf,1,env_sig)
            
            hilbert_pred = hilbert(y_s1[ch_ind,:]/np.abs(y_s1[ch_ind,:]).max())
            env_pred = np.abs(hilbert_pred)
            env_pred = signal.filtfilt(h_lpf,1,env_pred)
            
            coh = np.abs(np.fft.fft(env_sig) * np.fft.fft(env_pred)) ** 2 / (
                np.abs(np.fft.fft(env_sig))**2 * np.abs(np.fft.fft(env_pred))**2 )
            f = np.fft.fftfreq(coh.size,d=1/fs)
            
            f2,Cxy = signal.coherence(env_sig - env_sig.mean(),env_pred - env_pred.mean(),fs=fs, nfft=env_sig.size,detrend=None)
            
            axs[p1,p2].plot(f2,Cxy,color='r')
            axs[p1,p2].set_title(ch_picks[ch_ind])    

fig.suptitle('Coherence ')

fig,axs = plt.subplots(sbp2[0],sbp2[1],sharex=True,gridspec_kw=None)
for p1 in range(sbp2[0]):
    for p2 in range(sbp2[1]):
        cur_ch = p1*sbp2[1]+p2+sbp[0]*sbp[1]
        if np.any(cur_ch==ch_picks):
            ch_ind = np.where(cur_ch==ch_picks)[0][0]
            
            hilbert_sig = hilbert(sr1[cur_ch,:]/np.abs(sr1[cur_ch,:]).max())
            env_sig = np.abs(hilbert_sig)
            
            hilbert_pred = hilbert(y_s1[ch_ind,:]/np.abs(y_s1[ch_ind,:]).max())
            env_pred = np.abs(hilbert_pred)
            
            coh = np.abs(np.fft.fft(env_sig) * np.fft.fft(env_pred)) ** 2 / (
                np.abs(np.fft.fft(env_sig))**2 * np.abs(np.fft.fft(env_pred))**2 )
            f = np.fft.fftfreq(coh.size,d=1/fs)
            
            f2,Cxy = signal.coherence(env_sig - env_sig.mean(),env_pred - env_pred.mean(),fs=fs,nfft=env_sig.size,detrend=None)
            
            axs[p1,p2].plot(f2,Cxy,color='r')
            axs[p1,p2].set_title(ch_picks[ch_ind])   

fig.suptitle('Coherence')    



# plt.figure()
# plt.plot(t_sr1,sentEnv[0])




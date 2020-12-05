#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 13:04:01 2020

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


nchans = 34;
refchans = ['EXG1','EXG2']

data_eeg = [];
data_evnt = [];
  
#data_loc = '/media/ravinderjit/Storage2/EEGdata'
data_loc = '/media/ravinderjit/Data_Drive/Data/EEGdata/Neural_CMR'
subject = 'S211_plus9dB'
exclude = ['EXG3','EXG4','EXG5','EXG6','EXG7','EXG8']

datapath = os.path.join(data_loc,subject)

data_eeg,data_evnt = EEGconcatenateFolder(datapath+'/',nchans,refchans,exclude)
data_eeg.filter(l_freq=1,h_freq=300)

#%% Blink Removal
blinks = find_blinks(data_eeg,ch_name = ['A1'],thresh = 100e-6, l_trans_bandwidth = 0.5, l_freq =1.0)
blink_epochs = mne.Epochs(data_eeg,blinks,998,tmin=-0.25,tmax=0.25,proj=False,
                              baseline=(-0.25,0),reject=dict(eeg=500e-6))
Projs = compute_proj_epochs(blink_epochs,n_grad=0,n_mag=0,n_eeg=8,verbose='DEBUG')
blink_proj = [Projs[0]]

data_eeg.add_proj(blink_proj)
data_eeg.plot_projs_topomap()
data_eeg.plot(events=blinks,show_options=True)

#%% Plot 
#1 = 4 coh 0
#2 = 40 coh 0
#3 = 223 coh 0 
#4 = 4 coh 1
#5 = 40 coh 1
#6 = 223 coh 1
#7 = 2-10 coh 0
#8 = 2-10 coh 1
labels = ['4 coh 0', '40 coh 0', '223 coh 0', '4 coh 1','40 coh 1','223 coh 1',
          '2-10 coh 0','2-10 coh 1']
# labels = ['6 coh 0', '40 coh 0', '223 coh 0', '6 coh 1','40 coh 1','223 coh 1',
#            '6','40','223']

tmin = -0.5
tmax = 2.3
reject = dict(eeg=100e-6)
baseline = (-0.2,0)

epochs_1 = mne.Epochs(data_eeg,data_evnt,[1],tmin=tmin,tmax=tmax,
                              baseline=baseline,reject=reject)
epochs_2 = mne.Epochs(data_eeg,data_evnt,[2],tmin=tmin,tmax=tmax,
                              baseline=baseline,reject=reject)
epochs_3 = mne.Epochs(data_eeg,data_evnt,[3],tmin=tmin,tmax=tmax,
                              baseline=baseline,reject=reject)
epochs_4 = mne.Epochs(data_eeg,data_evnt,[4],tmin=tmin,tmax=tmax,
                              baseline=baseline,reject=reject)

epochs_5 = mne.Epochs(data_eeg,data_evnt,[5],tmin=tmin,tmax=tmax,
                              baseline=baseline,reject=reject)
epochs_6 = mne.Epochs(data_eeg,data_evnt,[6],tmin=tmin,tmax=tmax,
                              baseline=baseline,reject=reject)
epochs_7 = mne.Epochs(data_eeg,data_evnt,[7],tmin=tmin,tmax=tmax,
                              baseline=baseline,reject=reject)
epochs_8 = mne.Epochs(data_eeg,data_evnt,[8],tmin=tmin,tmax=tmax,
                              baseline=baseline,reject=reject)



evkd_1 = epochs_1.average()
evkd_2 = epochs_2.average()
evkd_3 = epochs_3.average()
evkd_4 = epochs_4.average()
evkd_5 = epochs_5.average()
evkd_6 = epochs_6.average()
evkd_7 = epochs_7.average()
evkd_8 = epochs_8.average()

picks = [4,30,31,7,22,8,21]
picks = 4
# evkd_1.plot(picks=picks,titles=labels[0])
# evkd_2.plot(picks=picks,titles=labels[1])
# evkd_3.plot(picks=picks,titles=labels[2])
# evkd_4.plot(picks=picks,titles=labels[3])
# evkd_5.plot(picks=picks,titles=labels[4])
# evkd_6.plot(picks=picks,titles=labels[5])
# evkd_7.plot(picks=picks,titles=labels[6])
# evkd_8.plot(picks=picks,titles=labels[7])
# evkd_9.plot(picks=picks,titles=labels[8])

#%% spectral analysievkd_1.plot(picks=picks,titles='4 coh 0')
fs = evkd_1.info['sfreq']

# data_1 = evkd_1.data
# data_2 = evkd_2.data
# data_3 = evkd_3.data
# data_4 = evkd_4.data
# data_5 = evkd_5.data
# data_6 = evkd_6.data
# data_7 = evkd_7.data
# data_8 = evkd_8.data
# data_9 = evkd_9.data

# nfft = 2**np.ceil(np.log(data_1[picks,:].size)/np.log(2))

# pxx = np.zeros((int(nfft/2),34,9))
# f,p1 = sa.periodogram(data_1.T, fs, nfft)
# pxx[:,:,0] = p1.squeeze()
# f,p1 = sa.periodogram(data_2.T, fs, nfft)
# pxx[:,:,1] = p1.squeeze()
# f,p1 = sa.periodogram(data_3.T, fs, nfft)
# pxx[:,:,2] = p1.squeeze()
# f,p1 = sa.periodogram(data_4.T, fs, nfft)
# pxx[:,:,3] = p1.squeeze()
# f,p1 = sa.periodogram(data_5.T, fs, nfft)
# pxx[:,:,4] = p1.squeeze()
# f,p1 = sa.periodogram(data_6.T, fs, nfft)
# pxx[:,:,5] = p1.squeeze()
# f,p1 = sa.periodogram(data_7.T, fs, nfft)
# pxx[:,:,6] = p1.squeeze()
# f,p1 = sa.periodogram(data_8.T, fs, nfft)
# pxx[:,:,7] = p1.squeeze()
# f,p1 = sa.periodogram(data_9.T, fs, nfft)
# pxx[:,:,8] = p1.squeeze()

    
# for j in np.arange(8):
#     plt.figure()
#     plt.plot(f,10*np.log10(pxx[:,:,j]))
#     plt.title(labels[j])
#     plt.legend(np.arange(32))
#     plt.plot(f,10*np.log10(pxx[j,:]))
#     plt.title(labels[j])


# stim_loc = '/media/ravinderjit/Data_Drive/Stim_Analysis/Stimuli/CMR/SNR_plus9/target_mods_9dB.mat'
# target_mods = sio.loadmat(stim_loc)
# target_mods = target_mods['target_mods_EEG'] #4,40,223,2-10

t = epochs_1.times
t1 = np.where(t>=0)[0][0]
t2 = np.where(t>=2.0)[0][0]
t = t[t1:t2]

target_mods = np.zeros((3,int(fs*2)))
# target_mods[0,:] = 0.5 + 0.5*np.sin(2*np.pi*6*t)
# target_mods[1,:] = 0.5 + 0.5*np.sin(2*np.pi*40*t)
# target_mods[2,:] = 0.5 + 0.5*np.sin(2*np.pi*223*t)
target_mods[0,:] = np.sin(2*np.pi*6*t)
target_mods[1,:] = np.sin(2*np.pi*40*t)
target_mods[2,:] = np.sin(2*np.pi*223*t)




dat_epochs_1 = epochs_1.get_data()
dat_epochs_2 = epochs_2.get_data()
dat_epochs_3 = epochs_3.get_data()
dat_epochs_4 = epochs_4.get_data()
dat_epochs_5 = epochs_5.get_data()
dat_epochs_6 = epochs_6.get_data()
dat_epochs_7 = epochs_7.get_data()
dat_epochs_8 = epochs_8.get_data()


dat_epochs_1 = dat_epochs_1[0:205,0:32,t1:t2].transpose(1,0,2)
dat_epochs_2 = dat_epochs_2[0:205,0:32,t1:t2].transpose(1,0,2)
dat_epochs_3 = dat_epochs_3[0:205,0:32,t1:t2].transpose(1,0,2)
dat_epochs_4 = dat_epochs_4[0:205,0:32,t1:t2].transpose(1,0,2)
dat_epochs_5 = dat_epochs_5[0:205,0:32,t1:t2].transpose(1,0,2)
dat_epochs_6 = dat_epochs_6[0:205,0:32,t1:t2].transpose(1,0,2)
dat_epochs_7 = dat_epochs_7[0:205,0:32,t1:t2].transpose(1,0,2)
dat_epochs_8 = dat_epochs_8[0:205,0:32,t1:t2].transpose(1,0,2)



params = dict()
params['Fs'] = fs
params['tapers'] = [2,2*2-1]
params['fpass'] = [1,300]
params['itc'] = 1


plvtap_1, f = mtplv(dat_epochs_1,params)
plvtap_2, f = mtplv(dat_epochs_2,params)
plvtap_3, f = mtplv(dat_epochs_3,params)
plvtap_4, f = mtplv(dat_epochs_4,params)
plvtap_5, f = mtplv(dat_epochs_5,params)
plvtap_6, f = mtplv(dat_epochs_6,params)
plvtap_7, f = mtplv(dat_epochs_7,params)
plvtap_8, f = mtplv(dat_epochs_8,params)


np.max(plvtap_5[:,75:85],axis=1)

plt.figure()
plt.plot(f,plvtap_1.T)
plt.title(labels[0])

plt.figure()
plt.plot(f,plvtap_2.T)
plt.title(labels[1])

plt.figure()
plt.plot(f,plvtap_3.T)
plt.title(labels[2])

plt.figure()
plt.plot(f,plvtap_4.T)
plt.title(labels[3])

plt.figure()
plt.plot(f,plvtap_5.T)
plt.title(labels[4])

plt.figure()
plt.plot(f,plvtap_6.T)
plt.title(labels[5])

plt.figure()
plt.plot(f,plvtap_7.T)
plt.title(labels[6])

plt.figure()
plt.plot(f,plvtap_8.T)
plt.title(labels[7])



# TW = 2
# Fres = (1/t[-1]) * TW * 2

# PLV_1, Coh_1, f = sa.PLV_Coh(target_mods[0,:], dat_epochs_1, TW, fs)
# PLV_2, Coh_2, f = sa.PLV_Coh(target_mods[1,:], dat_epochs_2, TW, fs)
# PLV_3, Coh_3, f = sa.PLV_Coh(target_mods[2,:], dat_epochs_3, TW, fs)
# PLV_4, Coh_4, f = sa.PLV_Coh(target_mods[0,:], dat_epochs_4, TW, fs)
# PLV_5, Coh_5, f = sa.PLV_Coh(target_mods[1,:], dat_epochs_5, TW, fs)
# PLV_6, Coh_6, f = sa.PLV_Coh(target_mods[2,:], dat_epochs_6, TW, fs)
# PLV_7, Coh_7, f = sa.PLV_Coh(target_mods[0,:], dat_epochs_7, TW, fs)
# PLV_8, Coh_8, f = sa.PLV_Coh(target_mods[1,:], dat_epochs_8, TW, fs)
# PLV_9, Coh_9, f = sa.PLV_Coh(target_mods[2,:], dat_epochs_9, TW, fs)

# Num_noiseFloors = 25
# Cohnf_1 = np.zeros([Coh_1.shape[0],Num_noiseFloors])
# Cohnf_2 = np.zeros([Coh_2.shape[0],Num_noiseFloors])
# Cohnf_3 = np.zeros([Coh_3.shape[0],Num_noiseFloors])
# Cohnf_4 = np.zeros([Coh_4.shape[0],Num_noiseFloors])
# Cohnf_5 = np.zeros([Coh_5.shape[0],Num_noiseFloors])
# Cohnf_6 = np.zeros([Coh_6.shape[0],Num_noiseFloors])
# Cohnf_7 = np.zeros([Coh_7.shape[0],Num_noiseFloors])
# Cohnf_8 = np.zeros([Coh_8.shape[0],Num_noiseFloors])
# Cohnf_9 = np.zeros([Coh_9.shape[0],Num_noiseFloors])

# for nf in range(0,Num_noiseFloors):
#     print('NF:',nf+1,'/',Num_noiseFloors)
#     data_nf = dat_epochs_1
#     targMod = target_mods[0,:]
#     order = np.random.permutation(data_nf.shape[1])
#     Y_1 = data_nf[:,order]
#     Y_1[:,0:int(np.round(order.size/2))] = - Y_1[:,0:int(np.round(order.size/2))]
#     PLVn_Y1, Cohn_Y1, f = sa.PLV_Coh(targMod,Y_1,TW,fs)
#     Cohnf_1[:,nf] = Cohn_Y1
    
#     data_nf = dat_epochs_2
#     targMod = target_mods[1,:]
#     order = np.random.permutation(data_nf.shape[1])
#     Y_1 = data_nf[:,order]
#     Y_1[:,0:int(np.round(order.size/2))] = - Y_1[:,0:int(np.round(order.size/2))]
#     PLVn_Y1, Cohn_Y1, f = sa.PLV_Coh(targMod,Y_1,TW,fs)
#     Cohnf_2[:,nf] = Cohn_Y1
    
#     data_nf = dat_epochs_3
#     targMod = target_mods[2,:]
#     order = np.random.permutation(data_nf.shape[1])
#     Y_1 = data_nf[:,order]
#     Y_1[:,0:int(np.round(order.size/2))] = - Y_1[:,0:int(np.round(order.size/2))]
#     PLVn_Y1, Cohn_Y1, f = sa.PLV_Coh(targMod,Y_1,TW,fs)
#     Cohnf_3[:,nf] = Cohn_Y1
    
#     data_nf = dat_epochs_4
#     targMod = target_mods[0,:]
#     order = np.random.permutation(data_nf.shape[1])
#     Y_1 = data_nf[:,order]
#     Y_1[:,0:int(np.round(order.size/2))] = - Y_1[:,0:int(np.round(order.size/2))]
#     PLVn_Y1, Cohn_Y1, f = sa.PLV_Coh(targMod,Y_1,TW,fs)
#     Cohnf_4[:,nf] = Cohn_Y1
    
#     data_nf = dat_epochs_5
#     targMod = target_mods[1,:]
#     order = np.random.permutation(data_nf.shape[1])
#     Y_1 = data_nf[:,order]
#     Y_1[:,0:int(np.round(order.size/2))] = - Y_1[:,0:int(np.round(order.size/2))]
#     PLVn_Y1, Cohn_Y1, f = sa.PLV_Coh(targMod,Y_1,TW,fs)
#     Cohnf_5[:,nf] = Cohn_Y1
    
#     data_nf = dat_epochs_6
#     targMod = target_mods[2,:]
#     order = np.random.permutation(data_nf.shape[1])
#     Y_1 = data_nf[:,order]
#     Y_1[:,0:int(np.round(order.size/2))] = - Y_1[:,0:int(np.round(order.size/2))]
#     PLVn_Y1, Cohn_Y1, f = sa.PLV_Coh(targMod,Y_1,TW,fs)
#     Cohnf_6[:,nf] = Cohn_Y1
    
#     data_nf = dat_epochs_7
#     targMod = target_mods[0,:]
#     order = np.random.permutation(data_nf.shape[1])
#     Y_1 = data_nf[:,order]
#     Y_1[:,0:int(np.round(order.size/2))] = - Y_1[:,0:int(np.round(order.size/2))]
#     PLVn_Y1, Cohn_Y1, f = sa.PLV_Coh(targMod,Y_1,TW,fs)
#     Cohnf_7[:,nf] = Cohn_Y1
    
#     data_nf = dat_epochs_8
#     targMod = target_mods[1,:]
#     order = np.random.permutation(data_nf.shape[1])
#     Y_1 = data_nf[:,order]
#     Y_1[:,0:int(np.round(order.size/2))] = - Y_1[:,0:int(np.round(order.size/2))]
#     PLVn_Y1, Cohn_Y1, f = sa.PLV_Coh(targMod,Y_1,TW,fs)
#     Cohnf_8[:,nf] = Cohn_Y1
    
#     data_nf = dat_epochs_8
#     targMod = target_mods[2,:]
#     order = np.random.permutation(data_nf.shape[1])
#     Y_1 = data_nf[:,order]
#     Y_1[:,0:int(np.round(order.size/2))] = - Y_1[:,0:int(np.round(order.size/2))]
#     PLVn_Y1, Cohn_Y1, f = sa.PLV_Coh(targMod,Y_1,TW,fs)
#     Cohnf_8[:,nf] = Cohn_Y1

    

# NF_4 = np.concatenate((Cohnf_1,Cohnf_4),axis=1)  
# NF_4_bot = NF_4.mean(axis=1) - 2*NF_4.std(axis=1)
# NF_4_top = NF_4.mean(axis=1) + 2*NF_4.std(axis=1)

# NF_40 = np.concatenate((Cohnf_2,Cohnf_5),axis=1)
# NF_40_bot = NF_40.mean(axis=1) - 2*NF_40.std(axis=1)
# NF_40_top = NF_40.mean(axis=1) + 2*NF_40.std(axis=1)    
    
# NF_223 = np.concatenate((Cohnf_3,Cohnf_6),axis=1)
# NF_223_bot = NF_223.mean(axis=1) - 2*NF_223.std(axis=1)
# NF_223_top = NF_223.mean(axis=1) + 2*NF_223.std(axis=1)   

# NF_2_10 = np.concatenate((Cohnf_7,Cohnf_8),axis=1)
# NF_2_10_bot = NF_2_10.mean(axis=1) - 2*NF_2_10.std(axis=1)
# NF_2_10_top = NF_2_10.mean(axis=1) + 2*NF_2_10.std(axis=1)   

# plt.figure()
# plt.plot(f,Coh_1)
# plt.plot(f,Coh_4)
# plt.plot(f,Coh_7)
# plt.plot(f,NF_4_bot,color='grey')
# plt.plot(f,NF_4_top,color='grey')
# plt.title('6')
# plt.legend(('Incoh','Coh','Mod Tone'))

# plt.figure()
# plt.plot(f,Coh_2)
# plt.plot(f,Coh_5)
# plt.plot(f,Coh_8)
# plt.plot(f,NF_40_bot,color='grey')
# plt.plot(f,NF_40_top,color='grey')
# plt.title('40')
# plt.legend(('Incoh','Coh','Mod Tone'))

# plt.figure()
# plt.plot(f,Coh_3)
# plt.plot(f,Coh_6)
# plt.plot(f,Coh_9)
# plt.plot(f,NF_223_bot,color='grey')
# plt.plot(f,NF_223_top,color='grey')
# plt.title('223')
# plt.legend(('Incoh','Coh','Mod Tone'))

# # plt.figure()
# # plt.plot(f,Coh_7)
# # plt.plot(f,Coh_8)
# # plt.plot(f,NF_2_10_bot,color='grey')
# # plt.plot(f,NF_2_10_top,color='grey')
# # plt.title('2-10')
# # plt.legend(('Incoh','Coh'))












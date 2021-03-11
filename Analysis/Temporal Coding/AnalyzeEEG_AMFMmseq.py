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
Mseq_loc = '/media/ravinderjit/Data_Drive/Data/EEGdata/TemporalCoding/mseqEEG_40_4096.mat'

Mseq_dat = sio.loadmat(Mseq_loc)
mseq = Mseq_dat['mseqEEG_4096']
mseq = mseq.astype(float)

#data_loc = '/media/ravinderjit/Storage2/EEGdata/'
data_loc = '/media/ravinderjit/Data_Drive/Data/EEGdata/TemporalCoding/AMFMmseq/'
subject = 'S211'

exclude = ['EXG3','EXG4','EXG5','EXG6','EXG7','EXG8']; #don't need these extra external channels that are saved


datapath =  os.path.join(data_loc, subject)
# datapath = '/media/ravinderjit/Data_Drive/Data/EEGdata/EFR'
data_eeg,data_evnt = EEGconcatenateFolder(datapath+'/',nchans,refchans,exclude)
data_eeg.filter(l_freq=1,h_freq=80)

#%% Blink Removal
blinks = find_blinks(data_eeg,ch_name = ['A1'],thresh = 100e-6, l_trans_bandwidth = 0.5, l_freq =1.0)
blink_epochs = mne.Epochs(data_eeg,blinks,998,tmin=-0.25,tmax=0.25,proj=False,
                              baseline=(-0.25,0),reject=dict(eeg=500e-6))
Projs = compute_proj_epochs(blink_epochs,n_grad=0,n_mag=0,n_eeg=8,verbose='DEBUG')
blink_proj = Projs[0]

data_eeg.add_proj(blink_proj)
# data_eeg.plot_projs_topomap()
# data_eeg.plot()

#%% Plot data

epoch_data_AM = mne.Epochs(data_eeg, data_evnt, [1], tmin=-0.3, tmax=13,reject=None, baseline=(-0.2, 0.)) 
evkd_data_AM= epoch_data_AM.average();
# evkd_data_AM.plot(picks = [31], titles = 'AMmseq')

# epoch_data_FM = mne.Epochs(data_eeg, data_evnt, [2], tmin=-0.3, tmax=13,reject=None, baseline=(-0.2, 0.)) 
# evkd_data_FM = epoch_data_FM.average();
# evkd_data_FM.plot(picks = [31], titles = 'FMmseq')

del data_eeg, data_evnt, evkd_data_AM #, evkd_data_FM

#%% Plot PSD

#%% Extract part of response when stim is on
t = epoch_data_AM.times
t1 = np.where(t>=0)[0][0]
t2 = t1 + mseq.size
AMdata = epoch_data_AM.get_data(picks=[31])
AMdata = AMdata.T[:,0,:]
AMdata = AMdata[t1:t2,:]
t = t[t1:t2]

# FMdata = epoch_data_FM.get_data(picks=[31])
# FMdata = FMdata.T[:,0,:]
# FMdata = FMdata[t1:t2,:]

#%% Remove epochs with large deflections
Peak2Peak = AMdata.max(axis=0) - AMdata.min(axis=0)
AMdata = AMdata[:,Peak2Peak*1e6 < 100.]

fs = epoch_data_AM.info['sfreq']
 
# Peak2Peak = FMdata.max(axis=0) - FMdata.min(axis=0)
# FMdata = FMdata[:,Peak2Peak*1e6<100.]

# AMcont = np.array([])
# MseqCont = np.array([])
# for k in range(AMdata.shape[1]):
#     AMcont = np.append(AMcont,AMdata[:,k])
#     MseqCont = np.append(MseqCont,mseq)

# AMcont = sp.signal.decimate(AMcont,10,zero_phase=True)
# MseqCont = sp.signal.decimate(MseqCont,10,zero_phase=True)
# AM_Ht = np.correlate(AMcont,MseqCont,mode='full')
# tt = np.arange(0,AM_Ht.size/(fs/10),1/(fs/10))
# plt.figure()
# plt.plot(tt[MseqCont.size-1:],AM_Ht[MseqCont.size-1:])

# AM_Ht_avg = np.correlate(AMdata.mean(axis=1),mseq[0,:],mode='full')
# AM_Ht_avg = AM_Ht_avg[mseq.size-1:]PLV_AM, Coh_AM, f, Phase_AM = PLV_Coh(mseq,AMdata,TW,fs)
# tt = np.arange(0,AM_Ht_avg.size/fs,1/fs)
# plt.figure()
# plt.plot(tt,AM_Ht_avg)

params = dict()
# AM_Ht_avg = np.expand_dims(AM_Ht_avg,1).T
params['Fs'] = epoch_data_AM.info['sfreq']
params['tapers'] = [2,2*2-1]
params['fpass'] = [0, 200]
params['itc'] = 0
# t_end = np.where(tt>=0.8)[0][0]
# AM_Hf_avg,f = mtspecraw(AM_Ht_avg[0:t_end],params)

X_data =  epoch_data_AM.get_data(picks=range(32))
X_data = X_data[:,:,t1:t2].transpose(1,0,2)
del epoch_data_AM
# AM_Hf_plv,f = mtplv(X_data[31,:,:],mseq,params,'DEBUG')

# plt.figure()
# plt.plot(f,AM_Hf_plv)

# plt.figure()
# plt.plot(f,10*np.log10(AM_Hf_avg))

TW = 12
Fres = (1/t[-1]) * TW * 2


PLV_AM, Coh_AM, f, Phase_AM = PLV_Coh(mseq,AMdata,TW,fs)
# PLV_FM, Coh_FM, f, Phase_FM = PLV_Coh(mseq,FMdata,TW,fs)

N = 2**np.log2(AMdata.shape[0])
AMdata_f = sp.fft(AMdata.mean(axis=1),axis=0,n=N)
mseq_f = sp.fft(mseq[0,:],axis=0,n=N)
ph_mseq = np.unwrap(np.angle(mseq_f))
ph_AM = np.unwrap(np.angle(AMdata_f)) - ph_mseq
f_ph = np.arange(0,fs-1./(N/fs),1./(N/fs))
plt.figure()
plt.plot(f_ph,ph_AM)


plt.figure()
plt.plot(f, Phase_AM)
plt.xlim([0,60])
plt.ylim([])

f2 = np.where(f>=4)[0][0]
f8 = np.where(f>=8)[0][0]

coeff = np.polyfit(f[f2:f8],Phase_AM[f2:f8,:],deg=1)
GD_line1 = coeff[0]*f[f2:f8] + coeff[1]
latency1 = -coeff[0] / (2*np.pi)

f17 = np.where(f>=18)[0][0]
f30 = np.where(f>=30)[0][0]

coeff2 = np.polyfit(f[f17:f30],Phase_AM[f17:f30],deg=1)
GD_line2 = coeff2[0]*f[f17:f30] + coeff2[1]
latency2 = -coeff2[0] / (2*np.pi)


fig = plt.figure()
plt.plot(f,Coh_AM,color='k')
plt.title('Coh AM')

# fig = plt.figure()
# plt.plot(f,Coh_FM,color='k')
# plt.title('Coh FM')

plt.figure()
plt.plot(f,PLV_AM)
plt.title('PLV AM')
plt.xlim([0,60])

plt.figure()
plt.plot(f, Phase_AM,color='k',linewidth=2)
plt.plot(f[f2:f8],GD_line1,color='red',linewidth=3)
plt.plot(f[f17:f30],GD_line2,color='blue',linewidth=3)
plt.title('Phase Response AM mseq')
plt.xlim([0,45])
#plt.ylim([-18,5])
plt.xlabel('Frequency (Hz)')
plt.ylabel('Radians')

# plt.figure()
# plt.plot(f, Phase_FM)
# plt.title('Phase FM')




#%% Noise floors
Num_noiseFloors = 20
Cohnf_AM = np.zeros([Coh_AM.shape[0],Num_noiseFloors])
PLVnf_AM = np.zeros([Coh_AM.shape[0],Num_noiseFloors])
# Cohnf_FM = np.zeros([Coh_FM.shape[0],Num_noiseFloors])

for nf in range(0,Num_noiseFloors):
    print('NF:',nf+1,'/',Num_noiseFloors)
    order_AM = np.random.permutation(AMdata.shape[1]-1)
    # order_FM = np.random.permutation(FMdata.shape[1]-1)
    Y_AM = AMdata[:,order_AM]
    # Y_FM = FMdata[:,order_AM]
    Y_AM[:,0:int(np.round(order_AM.size/2))] = -Y_AM[:,0:int(np.round(order_AM.size/2))]
    # Y_FM[:,0:int(np.round(order_FM.size/2))] = -Y_FM[:,0:int(np.round(order_FM.size/2))]
    
    PLVn_AM, Cohn_AM, f, Phase_AM_nf = PLV_Coh(mseq,Y_AM,TW,fs)
    # PLVn_FM, Cohn_FM, f = PLV_Coh(mseq,Y_FM,TW,fs)
    
    Cohnf_AM[:,nf] = Cohn_AM
    PLVnf_AM[:,nf] = PLVn_AM
    # Cohnf_FM[:,nf] = Cohn_FM
    

plt.figure()
plt.plot(f,Cohnf_AM,color='grey')
plt.plot(f,Coh_AM,color='k',linewidth=2)
plt.title('AM mseq Coherence')
plt.ylabel('Coherence')
plt.xlabel('Frequency (Hz)')
plt.xlim([0, 45])

# plt.figure()
# plt.plot(f,Cohnf_FM,color='grey')
# plt.plot(f,Coh_FM,color='k',linewidth=2)








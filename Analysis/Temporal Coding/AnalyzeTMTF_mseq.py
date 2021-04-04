# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import scipy.io as sio
import mne
from anlffr.preproc import find_blinks
from EEGpp import EEGconcatenateFolder
from mne.preprocessing.ssp import compute_proj_epochs
import os
import pickle

from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA

from anlffr.spectral import mtplv


nchans = 34;
refchans = ['EXG1','EXG2']

data_eeg = [];
data_evnt = [];


data_loc = '/media/ravinderjit/Data_Drive/Data/EEGdata/TemporalCoding/TMTF/'
Subjects = ['Hari 1']
subject = Subjects[0]

mseq_loc = os.path.join(data_loc, 'TMTFmseq_resampled.mat')
Mseq_dat = sio.loadmat(mseq_loc)
mseq = Mseq_dat['m'].astype('float')
mseq = 2*mseq - 1
mseq[mseq<0] = -1
mseq[mseq>0] =  1

plt.figure()
plt.plot(mseq)

# a = sp.fft(mseq[:,0])
# f = np.arange(0,fs,fs/a.shape[0])
# plt.figure()
# plt.plot(f,10*np.log10(np.abs(a)**2))




exclude = ['EXG3','EXG4','EXG5','EXG6','EXG7','EXG8']; #don't need these extra external channels that are saved
    
datapath =  os.path.join(data_loc, subject)
data_eeg,data_evnt = EEGconcatenateFolder(datapath+'/',nchans,refchans,exclude)
data_eeg.filter(l_freq=1,h_freq=1000)
#data_eeg, data_evnt = data_eeg.resample(4096,events=data_evnt)

#%% Blink Removal
blinks = find_blinks(data_eeg,ch_name = ['A1'],thresh = 100e-6, l_trans_bandwidth = 0.5, l_freq =1.0)
blink_epochs = mne.Epochs(data_eeg,blinks,998,tmin=-0.25,tmax=0.25,proj=False,
                              baseline=(-0.25,0),reject=dict(eeg=500e-6))
Projs = compute_proj_epochs(blink_epochs,n_grad=0,n_mag=0,n_eeg=8,verbose='DEBUG')

if subject ==  'Hari 1':
        ocular_projs = [Projs[0],Projs[1]]

data_eeg.add_proj(ocular_projs)
data_eeg.plot_projs_topomap()
data_eeg.plot(events=blinks,show_options=True)

del ocular_projs, blink_epochs, Projs, blinks

#%% Plot data and extract epochs
fs = data_eeg.info['sfreq']
reject = dict(eeg=150e-6)
epochs = []
# Evoked_resp = []
for j in range(2):
    epochs.append(mne.Epochs(data_eeg, data_evnt, [j+1], tmin=-0.5, 
                              tmax=np.ceil(mseq.size/fs),reject=reject, reject_tmin = 0, reject_tmax=mseq.size/fs, baseline=(-0.2, 0.)) )
    # Evoked_resp.append(epochs[j].average())
    #Evoked_resp[j]
    epochs[j].average().plot(picks=[31],titles=str(j))
    


    



#%% Extract part of response when stim is on
ch_picks = np.arange(32)
remove_chs = []
ch_picks = np.delete(ch_picks,remove_chs)

epdat = []
evkdat = []
fs = epochs[0].info['sfreq']
t = epochs[0].times
# fs = Evoked_resp[0].info['sfreq']
# t = Evoked_resp[0].times
t1 = np.where(t>=0)[0][0]
t2 = t1 + mseq.size
t = t[t1:t2]  
for m in range(len(epochs)):
# for m in range(len(Evoked_resp)):
    epdat.append(epochs[m].get_data(picks=ch_picks)[:,ch_picks,t1:t2].transpose(1,0,2))
    # evkdat.append(Evoked_resp[m].data[ch_picks,t1:t2])



#%% Remove epochs with large deflections
Reject_Thresh=150e-6
Tot_trials = np.zeros([len(mseq)])
for m in range(len(epochs)):
    Peak2Peak = epdat[m].max(axis=2) - epdat[m].min(axis=2)
    mask_trials = np.all(Peak2Peak <Reject_Thresh,axis=0)
    print('rejected ' + str(epdat[m].shape[1] - sum(mask_trials)) + ' trials due to P2P')
    epdat[m] = epdat[m][:,mask_trials,:]
    print('Total Trials Left: ' + str(epdat[m].shape[1]))
    Tot_trials[m] = epdat[m].shape[1]
    
plt.figure()
plt.plot(Peak2Peak.T)


#%% compute PLV
TW = 6
Fres = (1/t[-1]) * TW * 2

params = dict()
params['Fs'] = fs
params['tapers'] = [TW,2*TW-1]
params['fpass'] = [0, 1000]
params['itc'] = 0

plv_m=[]
for m in range(len(epdat)):
    print('On mseq # ' + str(m+1))
    plv,f = mtplv(epdat[m][30:,:,:],params)
    plv_m.append(plv)
    
plt.figure()
plt.plot(f,plv_m[1].T)
    

#%% Correlation Analysis
    
tend = 1.0 #time of Ht to keep
tend_ind = round(tend*fs) - 1


Ht = []
Htnf = []
# do cross corr
for m in range(len(epdat)): 
# for m in range(len(evkdat)): 
    print('On mseq # ' + str(m+1))
    resp_m = epdat[m].mean(axis=1)
    # resp_m = evkdat[m]
    Ht_m = np.zeros(resp_m.shape)
    for ch in range(resp_m.shape[0]):
        Ht_m[ch,:] = np.correlate(resp_m[ch,:],mseq[:,0],mode='full')[mseq.size-1:]
    Ht.append(Ht_m)
    # for nf in range(num_nfs):
    #     resp = epdat[m]
    #     inv_inds = np.random.permutation(epdat[m].shape[1])[:round(epdat[m].shape[1]/2)]
    #     resp[:,inv_inds,:] = -resp[:,inv_inds,:]
    #     resp_nf = resp.mean(axis=1)
    #     Ht_nf = np.zeros(resp_nf.shape)
    #     for ch in range(resp_nf.shape[0]):
    #         Ht_nf[ch,:] = np.correlate(resp_nf[ch,:],mseq[m][0,:],mode='full')[mseq[m].size-1:]
    #     Htnf.append(Ht_nf[:,:tend_ind])
    
#only keep Ht up to tend 
for h in range(len(Ht)):
    Ht[h] = Ht[h][:,:tend_ind]
t = t[:tend_ind]

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
        


for m in range(len(Ht)):
    Ht_1 = Ht[m]
    fig,axs = plt.subplots(sbp[0],sbp[1],sharex=True,gridspec_kw=None)
    for p1 in range(sbp[0]):
        for p2 in range(sbp[1]):
            axs[p1,p2].plot(t,Ht_1[p1*sbp[1]+p2,:],color='k')
            axs[p1,p2].set_title(ch_picks[p1*sbp[1]+p2])    
            # for n in range(m*num_nfs,num_nfs*(m+1)):
            #     axs[p1,p2].plot(t,Htnf[n][p1*sbp[1]+p2,:],color='grey',alpha=0.3)
            
    fig.suptitle('Ht ' + str(m) )
    
    
    fig,axs = plt.subplots(sbp2[0],sbp2[1],sharex=True,gridspec_kw=None)
    for p1 in range(sbp2[0]):
        for p2 in range(sbp2[1]):
            axs[p1,p2].plot(t,Ht_1[p1*sbp2[1]+p2+sbp[0]*sbp[1],:],color='k')
            axs[p1,p2].set_title(ch_picks[p1*sbp2[1]+p2+sbp[0]*sbp[1]])   
            # for n in range(m*num_nfs,num_nfs*(m+1)):
            #     axs[p1,p2].plot(t,Htnf[n][p1*sbp[1]+p2,:],color='grey',alpha=0.3)
            
    fig.suptitle('Ht ' + str(m) )    
        
#%% PCA decomposition of Ht
pca_sp = []
pca_coeff = []
pca_expVar = []

pca_sp_nf = []
pca_coeff_nf = []
pca_expVar_nf = []

n_comp = 2

for m in range(len(Ht)):
    pca = PCA(n_components=n_comp)
    pca.fit(Ht[m])
    pca_space = pca.fit_transform(Ht[m].T)
    
   
    
    pca_sp.append(pca_space)
    pca_coeff.append(pca.components_)
    pca_expVar.append(pca.explained_variance_ratio_)
    
# for n in range(len(Htnf)):
#     pca = PCA(n_components=n_comp)
#     pca.fit(Htnf[n])
#     pca_space = pca.fit_transform(Htnf[n].T)

#     pca_sp_nf.append(pca_space)
#     pca_coeff_nf.append(pca.components_)
#     pca_expVar_nf.append(pca.explained_variance_ratio_)
    
for m in range(len(pca_sp)):
    fig,axs = plt.subplots(2,1)
    axs[0].plot(t,pca_sp[m])
    # for n in range(m*num_nfs,num_nfs*(m+1)):
    #     axs[0].plot(t,pca_sp_nf[n],color='grey',alpha=0.3)
    
    
    axs[1].plot(ch_picks,pca_coeff[m].T)
    axs[1].set_xlabel('channel')
    # for n in range(m*num_nfs,num_nfs*(m+1)):
    #     axs[1].plot(ch_picks,pca_coeff_nf[n].T,color='grey',alpha=0.1)
    fig.suptitle('PCA ' + str(m))    
    
p_ind = 1
vmin = pca_coeff[p_ind].mean() - 2 * pca_coeff[p_ind].std()
vmax = pca_coeff[p_ind].mean() + 2 * pca_coeff[p_ind].std()
plt.figure()
mne.viz.plot_topomap(pca_coeff[p_ind][0,:], mne.pick_info(epochs[0].info, ch_picks),vmin=vmin,vmax=vmax)
plt.figure()
mne.viz.plot_topomap(pca_coeff[p_ind][1,:], mne.pick_info(epochs[0].info, ch_picks),vmin=vmin,vmax=vmax)


           

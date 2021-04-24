#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 13:54:47 2021

@author: ravinderjit
"""

import os
import pickle
import mne
import numpy as np
import scipy as sp
from mne.preprocessing.ssp import compute_proj_epochs
import scipy.io as sio
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from EEGpp import EEGconcatenateFolder
from anlffr.preproc import find_blinks
import random

file_loc = '/media/ravinderjit/Data_Drive/Data/EEGdata/TemporalCoding/mseqEEG_150_bits10_4096.mat' 
Mseq_dat = sio.loadmat(file_loc)
mseq = Mseq_dat['mseqEEG_4096'].astype(float) 

fs =4096
t_m = np.arange(0,mseq.size/fs,1/fs)

plt.figure()
plt.plot(t_m,mseq.T)




#%% ACR

subject = 'S211'
pickle_loc = '/media/ravinderjit/Data_Drive/Data/EEGdata/TemporalCoding/AMmseq_bits4/Pickles_full/'

with open(os.path.join(pickle_loc,subject+'_AMmseqbits4.pickle'),'rb') as file:
    [tdat, Tot_trials, Ht, Htnf, info_obj, ch_picks]= pickle.load(file)


sbp = [4,4]
sbp2 = [4,4]

fig,axs = plt.subplots(sbp[0],sbp[1],sharex=True,sharey=True)
t=tdat[3]
Ht_1 = Ht[3]
for p1 in range(sbp[0]):
    for p2 in range(sbp[1]):
        cur_ch = p1*sbp[1]+p2
        if np.any(cur_ch==ch_picks):
            ch_ind = np.where(cur_ch==ch_picks)[0][0]
            axs[p1,p2].plot(t,Ht_1[ch_ind,:])
            axs[p1,p2].set_title(ch_picks[ch_ind])    
            # axs[p1,p2].set_xlim([0,0.5])


fig,axs = plt.subplots(sbp[0],sbp[1],sharex=True,sharey=True)
for p1 in range(sbp2[0]):
    for p2 in range(sbp2[1]):
        cur_ch = p1*sbp2[1]+p2+sbp[0]*sbp[1]
        if np.any(cur_ch==ch_picks):
            ch_ind = np.where(cur_ch==ch_picks)[0][0]
            axs[p1,p2].plot(t,Ht_1[ch_ind,:])
            axs[p1,p2].set_title(ch_picks[ch_ind])  
            
            
#%% Load EEG data
EEGdata_loc = '/media/ravinderjit/Data_Drive/Data/EEGdata/TemporalCoding/AMmseq_bits4/'
subject = 'S211'
refchans = ['EXG1','EXG2']

exclude = ['EXG3','EXG4','EXG5','EXG6','EXG7','EXG8']; #don't need these extra external channels that are saved
nchans = 34;
datapath =  os.path.join(EEGdata_loc, subject)

data_eeg,data_evnt = EEGconcatenateFolder(datapath+'/',nchans,refchans,exclude)
data_eeg.filter(l_freq=1,h_freq=1000)

blinks = find_blinks(data_eeg,ch_name = ['A1'],thresh = 100e-6, l_trans_bandwidth = 0.5, l_freq =1.0)
blink_epochs = mne.Epochs(data_eeg,blinks,998,tmin=-0.25,tmax=0.25,proj=False,
                              baseline=(-0.25,0),reject=dict(eeg=500e-6))
Projs = compute_proj_epochs(blink_epochs,n_grad=0,n_mag=0,n_eeg=8)

ocular_projs = [Projs[0]]
data_eeg.add_proj(ocular_projs)
data_eeg.plot_projs_topomap()

rand_evnt_pts = np.zeros((295,3,),dtype=int)
for r in np.arange(295):
    rand_evnt_pts[r,:] = [random.randint(data_evnt[0,0],data_evnt[-1,0]),int(0),int(5)]

data_evnt = np.concatenate([data_evnt,rand_evnt_pts])

reject = dict(eeg=800e-6)
epochs = []
labels = ['7bits', '8bits','9bits','10bits']
epochs = (mne.Epochs(data_eeg, data_evnt, 4, tmin=-0.3, 
     tmax=6.8+0.4,reject=reject, baseline=(-0.2, 0.)) )
epochs_rand = (mne.Epochs(data_eeg, data_evnt, 5, tmin=-0.3, 
      tmax=6.8+0.4,reject=reject, baseline=(-0.2, 0.)) )

epochs.average().plot_topo()
epochs_rand.average().plot_topo()

#%% Figure Out PAM peak
t = epochs.times
t1 = np.where(t>=0)[0][0]
t2 = t1 + mseq.size 
t= t[t1:t2]
chA11_pam = epochs.get_data(picks='A11')[:,0,t1:t2].T
chA11_rand = epochs_rand.get_data(picks='A11')[:,0,t1:t2].T

chA11_pam = chA11_pam.mean(axis=1)
chA11_pam = chA11_pam - chA11_pam.mean()
chA11_rand = chA11_rand.mean(axis=1)
chA11_rand = chA11_rand - chA11_rand.mean()

plt.figure()
plt.plot(t,chA11_pam)
plt.plot(t,mseq[0,:]*4e-6)
plt.title('A11')

plt.figure()
plt.plot(t,chA11_rand)
#plt.plot(t,chA11_pam)
plt.title('A11 rand')


t_xcorr = np.concatenate((-t[-1:0:-1],t))

A11_Ht = np.correlate(chA11_pam,mseq[0,:],mode='full')
plt.figure()
plt.plot(t_xcorr,A11_Ht)
plt.title('Xcorr of A11 and mseq')

A11_acorr = np.correlate(chA11_pam,chA11_pam,mode='full')
plt.figure()
plt.plot(t_xcorr,A11_acorr)
plt.title('A11 AutoCorr')

A10rand_Ht = np.correlate(chA11_rand,mseq[0,:],mode='full')
plt.figure()
plt.plot(t_xcorr,A10rand_Ht)
plt.title('Xcorr of A11_rand and mseq')

A10rand_acorr = np.correlate(chA11_rand,chA11_rand,mode='full')
plt.figure()
plt.plot(t_xcorr,A10rand_acorr)
plt.title('A11_rand AutoCorr')

mseq_part = mseq[0,:int(round(fs*2))]
mseq_test = np.concatenate([mseq_part,np.zeros(mseq.size-mseq_part.size)])

mod_mseq_xcorr = np.correlate(chA11_pam,mseq_test,mode='full')
plt.figure()
plt.plot(t_xcorr,mod_mseq_xcorr)
plt.plot(t_xcorr,A11_Ht)
plt.title('Mseq test')


# data_dict = {
#     'chA10': chA11_pam,
#     't':t,
#     'fs':fs,
#     'mseq':mseq,
#     }

# sp.io.savemat('NonCausal.mat',data_dict)









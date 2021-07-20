#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 17 19:07:52 2021

@author: ravinderjit
"""

import os
import pickle
import numpy as np
import mne
import matplotlib.pyplot as plt
from scipy.signal import freqz
from sklearn.decomposition import PCA
from sklearn.metrics import explained_variance_score
import scipy.io as sio


mseq_loc = '/media/ravinderjit/Data_Drive/Data/EEGdata/TemporalCoding/mseqEEG_150_bits10_4096.mat'
Mseq_dat = sio.loadmat(mseq_loc)
mseq = Mseq_dat['mseqEEG_4096'].astype(float)

data_loc = '/media/ravinderjit/Data_Drive/Data/EEGdata/TemporalCoding/AMmseq_10bits/'
pickle_loc = data_loc + 'Pickles/'

Subjects = ['S207', 'S228','S236','S238','S239','S246','S247','S250']

fs = 4096

A_Tot_trials = []
A_Ht = []
A_Htnf = []
A_info_obj = []
A_ch_picks = []

#%% Load Data

for sub in range(len(Subjects)):
    subject = Subjects[sub]
    with open(os.path.join(pickle_loc,subject +'_AMmseq10bits.pickle'),'rb') as file:
        [t, Tot_trials, Ht, Htnf, info_obj, ch_picks] = pickle.load(file)
        
    A_Tot_trials.append(Tot_trials)
    A_Ht.append(Ht)
    A_Htnf.append(Htnf)
    A_info_obj.append(info_obj)
    A_ch_picks.append(ch_picks)
    
#%% Get average Ht
perCh = np.zeros([32,1])
for s in range(len(Subjects)):
    for ch in range(32):
        perCh[ch,0] += np.sum(A_ch_picks[s]==ch)

Avg_Ht = np.zeros([32,t.size])
for s in range(len(Subjects)):
    Avg_Ht[A_ch_picks[s],:] += A_Ht[s]

Avg_Ht = Avg_Ht / perCh


#%% Plot time domain Ht

num_nf = len(A_Htnf[0])

sbp = [4,4]
sbp2 = [4,4]

fig,axs = plt.subplots(sbp[0],sbp[1],sharex=True)
for sub in range(len(A_Ht) + 1):
    if sub == len(A_Ht):
        Ht_1 = Avg_Ht
        ch_picks_s = np.arange(32)
    else:
        Ht_1 = A_Ht[sub]
        ch_picks_s = A_ch_picks[sub]
    for p1 in range(sbp[0]):
        for p2 in range(sbp[1]):
            cur_ch = p1*sbp[1] + p2
            if np.any(cur_ch==ch_picks_s):
                ch_ind = np.where(cur_ch==ch_picks_s)[0][0]
                if sub == len(A_Ht):
                    axs[p1,p2].plot(t,Ht_1[ch_ind,:],color='k',linewidth=2)
                    axs[p1,p2].set_title(ch_picks[ch_ind])
                    axs[p1,p2].set_xlim([-0.050,0.5])
                else:
                    axs[p1,p2].plot(t,Ht_1[ch_ind,:])
                    axs[p1,p2].set_title(ch_picks[ch_ind])
                    axs[p1,p2].set_xlim([-0.050,0.5])
                    
plt.legend(Subjects)
                
    
fig,axs = plt.subplots(sbp[0],sbp[1],sharex=True)
for sub in range(len(A_Ht)+1):
    if sub == len(A_Ht):
        Ht_1 = Avg_Ht
        ch_picks_s = np.arange(32)
    else:
        Ht_1 = A_Ht[sub]
        ch_picks_s = A_ch_picks[sub]
    for p1 in range(sbp[0]):
        for p2 in range(sbp[1]):
            cur_ch = p1*sbp2[1]+p2+sbp[0]*sbp[1]
            if np.any(cur_ch==ch_picks_s):
                ch_ind = np.where(cur_ch==ch_picks_s)[0][0]
                if sub == len(A_Ht):
                    axs[p1,p2].plot(t,Ht_1[ch_ind,:],color='k',linewidth=2)
                    axs[p1,p2].set_title(ch_picks[ch_ind])
                    axs[p1,p2].set_xlim([-0.050,0.5])
                else:
                    axs[p1,p2].plot(t,Ht_1[ch_ind,:])
                    axs[p1,p2].set_title(ch_picks[ch_ind])
                    axs[p1,p2].set_xlim([-0.050,0.5])
                
                   
#%% PCA on t_cuts

t_cuts = [.016, .066,.250,0.500]

pca_sp_cuts = []
pca_expVar_cuts = []
pca_coeff_cuts = []
pca_expVar2s = []

t_cutT = []
pca = PCA(n_components=2)

for t_c in range(len(t_cuts)):
    if t_c ==0:
        t_1 = np.where(t>=0)[0][0]
    else:
        t_1 = np.where(t>=t_cuts[t_c-1])[0][0]
    
    t_2 = np.where(t>=t_cuts[t_c])[0][0]
    pca_sp = pca.fit_transform(Avg_Ht[:,t_1:t_2].T)
    pca_expVar = pca.explained_variance_ratio_
    pca_coeff = pca.components_
    
    if pca_coeff[0,31] < 0:  # Expand this too look at mutlitple electrodes
       pca_coeff = -pca_coeff
       pca_sp = -pca_sp
       
    Avg_demean = Avg_Ht[:,t_1:t_2] - Avg_Ht[:,t_1:t_2].mean(axis=1)[:,np.newaxis]
    H_tc_est = np.matmul(pca_sp[:,0][:,np.newaxis],pca_coeff[0,:][np.newaxis,:])
    pca_expVar2 = explained_variance_score(Avg_demean.T, H_tc_est,multioutput='variance_weighted')
    
    pca_expVar2s.append(pca_expVar2)
    pca_sp_cuts.append(pca_sp)
    pca_expVar_cuts.append(pca_expVar)
    pca_coeff_cuts.append(pca_coeff)
    t_cutT.append(t[t_1:t_2])
    
    plt.figure()
    plt.plot(t[t_1:t_2], Avg_demean[31,:])
    plt.plot(t[t_1:t_2], H_tc_est[:,31])
    
    
    

plt.figure()
for t_c in range(len(t_cuts)):
    plt.plot(t_cutT[t_c],pca_sp_cuts[t_c][:,0])
    
plt.figure()
plt.title('2nd component')
for t_c in range(len(t_cuts)):
    plt.plot(t_cutT[t_c],pca_sp_cuts[t_c][:,1])
    
plt.figure()
plt.plot(t,Avg_Ht[31,:])
plt.xlim([0,0.5])

plt.figure()
labels = ['comp1', 'comp2']
vmin = pca_coeff_cuts[-1][0,:].mean() - 2 * pca_coeff_cuts[-1][0,:].std()
vmax = pca_coeff_cuts[-1][0,:].mean() + 2 * pca_coeff_cuts[-1][0,:].std()
for t_c in range(len(t_cuts)):
    plt.subplot(2,len(t_cuts),t_c+1)
    plt.title('ExpVar ' + str(np.round(pca_expVar_cuts[t_c][0]*100)) + '%')
    mne.viz.plot_topomap(pca_coeff_cuts[t_c][0,:], mne.pick_info(A_info_obj[1],A_ch_picks[1]),vmin=vmin,vmax=vmax)
    
    plt.subplot(2,len(t_cuts),t_c+1 + len(t_cuts))
    plt.title('ExpVar ' + str(np.round(pca_expVar_cuts[t_c][1]*100)) + '%')
    mne.viz.plot_topomap(pca_coeff_cuts[t_c][1,:], mne.pick_info(A_info_obj[1],A_ch_picks[1]),vmin=vmin,vmax=vmax)
    
    
#%% Use PCA template on t_splits for individuals

pca_sp_cuts_sub = []
pca_expVar_cuts_sub = []

for sub in range(len(Subjects)):
    pca_sp_cuts_ = []
    pca_expVar_cuts_ = []

    H_t = A_Ht[sub].T
    for t_c in range(len(t_cuts)):
        if t_c ==0:
            t_1 = np.where(t>=0)[0][0]
        else:
            t_1 = np.where(t>=t_cuts[t_c-1])[0][0]
        
        t_2 = np.where(t>=t_cuts[t_c])[0][0]
        
        H_tc = H_t[t_1:t_2,:]- H_t[t_1:t_2,:].mean(axis=0)[np.newaxis,:]
        
        pca_sp = np.matmul(H_tc,pca_coeff_cuts[t_c][0,A_ch_picks[sub]])
        
        H_tc_est = np.matmul(pca_coeff_cuts[t_c][0,A_ch_picks[sub]][:,np.newaxis],pca_sp[np.newaxis,:])
        pca_expVar = explained_variance_score(H_tc,H_tc_est.T, multioutput='variance_weighted')
        
        pca_sp_cuts_.append(pca_sp)
        pca_expVar_cuts_.append(pca_expVar)
    
    pca_sp_cuts_sub.append(pca_sp_cuts_)
    pca_expVar_cuts_sub.append(pca_expVar_cuts_)
    
    
for t_c in range(len(t_cuts)):
    plt.figure()
    for sub in range(len(Subjects)):
        plt.plot(t_cutT[t_c],pca_sp_cuts_sub[sub][t_c]/np.max(pca_sp_cuts_sub[sub][0]))
    
    
#%% Load data collected earlier
        
data_loc_old = '/media/ravinderjit/Data_Drive/Data/EEGdata/TemporalCoding/AMmseq_bits4/'
pickle_loc_old = data_loc_old + 'Pickles_full/'
Subjects_old = ['S211','S207','S236','S228','S238'] 

A_Tot_trials_old = []
A_Ht_old = []
A_Htnf_old = []
A_info_obj_old = []
A_ch_picks_old = []

for sub in range(len(Subjects_old)):
    subject = Subjects_old[sub]
    with open(os.path.join(pickle_loc_old,subject+'_AMmseqbits4.pickle'),'rb') as file:
        # [tdat, Tot_trials, Ht, Htnf, pca_sp, pca_coeff, pca_expVar, 
        #  pca_sp_nf, pca_coeff_nf,pca_expVar_nf,ica_sp,
        #  ica_coeff,ica_sp_nf,ica_coeff_nf, info_obj, ch_picks] = pickle.load(file)
        [tdat, Tot_trials, Ht, Htnf, info_obj, ch_picks] = pickle.load(file)
    
    A_Tot_trials_old.append(Tot_trials[3])
    A_Ht_old.append(Ht[3])
    A_Htnf_old.append(Htnf[3])
    
    A_info_obj_old.append(info_obj)
    A_ch_picks_old.append(ch_picks)
    
t = tdat[3]
    
#%% Avg Ht on old data

perCh = np.zeros([32,1])
for s in range(len(Subjects_old)):
    for ch in range(32):
        perCh[ch,0] += np.sum(A_ch_picks_old[s]==ch)

Avg_Ht_old = np.zeros([32,t.size])
for s in range(len(Subjects_old)):
    Avg_Ht_old[A_ch_picks_old[s],:] += A_Ht_old[s]

Avg_Ht_old = Avg_Ht_old / perCh

#%% PCA on old data

pca_sp_cuts_old = []
pca_expVar_cuts_old = []
pca_coeff_cuts_old = []

for t_c in range(len(t_cuts)):
    if t_c ==0:
        t_1 = np.where(t>=0)[0][0]
    else:
        t_1 = np.where(t>=t_cuts[t_c-1])[0][0]
    
    t_2 = np.where(t>=t_cuts[t_c])[0][0]
    pca_sp = pca.fit_transform(Avg_Ht_old[:,t_1:t_2].T)
    pca_expVar = pca.explained_variance_ratio_
    pca_coeff = pca.components_
    
    if pca_coeff[0,31] < 0:  # Expand this too look at mutlitple electrodes
       pca_coeff = -pca_coeff
       pca_sp = -pca_sp
       
    pca_sp_cuts_old.append(pca_sp)
    pca_expVar_cuts_old.append(pca_expVar)
    pca_coeff_cuts_old.append(pca_coeff)
     

plt.figure()
plt.plot(pca_coeff_cuts[0][0,:])
plt.plot(pca_coeff_cuts_old[0][0,:])

plt.figure()
plt.plot(pca_coeff_cuts[1][0,:])
plt.plot(pca_coeff_cuts_old[1][0,:])

#%% PCA on old individual data

pca_sp_cuts_sub_old = []
pca_expVar_cuts_sub_old = []

for sub in range(len(Subjects_old)):
    pca_sp_cuts_ = []
    pca_expVar_cuts_ = []
    H_t = A_Ht_old[sub].T
    for t_c in range(len(t_cuts)):
        if t_c ==0:
            t_1 = np.where(t>=0)[0][0]
        else:
            t_1 = np.where(t>=t_cuts[t_c-1])[0][0]
        
        t_2 = np.where(t>=t_cuts[t_c])[0][0]
        
        H_tc = H_t[t_1:t_2,:]- H_t[t_1:t_2,:].mean(axis=0)[np.newaxis,:]
        
        pca_sp = np.matmul(H_tc,pca_coeff_cuts[t_c][0,A_ch_picks_old[sub]])
        
        H_tc_est = np.matmul(pca_coeff_cuts[t_c][0,A_ch_picks_old[sub]][:,np.newaxis],pca_sp[np.newaxis,:])
        pca_expVar = explained_variance_score(H_tc,H_tc_est.T, multioutput='variance_weighted')
        
        pca_sp_cuts_.append(pca_sp)
        pca_expVar_cuts_.append(pca_expVar)
    
    pca_sp_cuts_sub_old.append(pca_sp_cuts_)
    pca_expVar_cuts_sub_old.append(pca_expVar_cuts_)
    
    
for t_c in range(len(t_cuts)):
    plt.figure()
    for sub in range(len(Subjects_old)):
        plt.plot(t_cutT[t_c],pca_sp_cuts_sub_old[sub][t_c]/np.max(pca_sp_cuts_sub_old[sub][0]))
    

#%% Compare old and new

for sub in range(1,len(Subjects_old)):
    index_new = Subjects.index(Subjects_old[sub])
    
    plt.figure()
    plt.title(Subjects_old[sub])
    plt.plot(t_cutT[0],pca_sp_cuts_sub_old[sub][0]/np.max(pca_sp_cuts_sub_old[sub][0]),color='tab:blue')
    plt.plot(t_cutT[1],pca_sp_cuts_sub_old[sub][1]/np.max(pca_sp_cuts_sub_old[sub][0]),color='tab:blue')
    
    plt.plot(t_cutT[0],pca_sp_cuts_sub[index_new][0]/np.max(pca_sp_cuts_sub[index_new][0]),color='tab:orange')
    plt.plot(t_cutT[1],pca_sp_cuts_sub[index_new][1]/np.max(pca_sp_cuts_sub[index_new][0]),color='tab:orange')
    
    
#Plot Cz and frontal electrodes

#Plot freq responses



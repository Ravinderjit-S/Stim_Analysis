#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 11:20:23 2021

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

dataActive_loc = '/media/ravinderjit/Data_Drive/Data/EEGdata/TemporalCoding/AMmseq_active/'
pickle_loc = dataActive_loc + 'Pickles/'

Subjects = ['S207', 'S228','S236','S238','S239','S246','S247','S250']

fs = 4096

A_Tot_trials_act = []
A_Ht_act = []
A_Htnf_act = []
A_info_obj_act = []
A_ch_picks_act = []


#%% Load Data

for sub in range(len(Subjects)):
    subject = Subjects[sub]
    with open(os.path.join(pickle_loc,subject +'_AMmseq10bits_Active.pickle'),'rb') as file:
        [t, Tot_trials, Ht, Htnf, info_obj, ch_picks] = pickle.load(file)
        
    A_Tot_trials_act.append(Tot_trials)
    A_Ht_act.append(Ht)
    A_Htnf_act.append(Htnf)
    A_info_obj_act.append(info_obj)
    A_ch_picks_act.append(ch_picks)
    
#%% Get average Ht
    
perCh = np.zeros([32,1])
for s in range(len(Subjects)):
    for ch in range(32):
        perCh[ch,0] += np.sum(A_ch_picks_act[s]==ch)

Avg_Ht_act = np.zeros([32,t.size])
for s in range(len(Subjects)):
    Avg_Ht_act[A_ch_picks_act[s],:] += A_Ht_act[s]

Avg_Ht_act = Avg_Ht_act / perCh
    
#%% PCA on t_cuts

t_cuts = [.016, .066,.250,0.500]

pca_sp_cuts_act = []
pca_expVar_cuts_act = []
pca_coeff_cuts_act = []

t_cutT = []
pca = PCA(n_components=2)

for t_c in range(len(t_cuts)):
    if t_c ==0:
        t_1 = np.where(t>=0)[0][0]
    else:
        t_1 = np.where(t>=t_cuts[t_c-1])[0][0]
    
    t_2 = np.where(t>=t_cuts[t_c])[0][0]
    pca_sp = pca.fit_transform(Avg_Ht_act[:,t_1:t_2].T)
    pca_expVar = pca.explained_variance_ratio_
    pca_coeff = pca.components_
    
    if pca_coeff[0,31] < 0:  # Expand this too look at mutlitple electrodes
       pca_coeff = -pca_coeff
       pca_sp = -pca_sp
       
    
    pca_sp_cuts_act.append(pca_sp)
    pca_expVar_cuts_act.append(pca_expVar)
    pca_coeff_cuts_act.append(pca_coeff)
    t_cutT.append(t[t_1:t_2])

plt.figure()
for t_c in range(len(t_cuts)):
    plt.plot(t_cutT[t_c],pca_sp_cuts_act[t_c][:,0])
    
plt.figure()
plt.title('2nd component')
for t_c in range(len(t_cuts)):
    plt.plot(t_cutT[t_c],pca_sp_cuts_act[t_c][:,1])    

plt.figure()
plt.plot(t,Avg_Ht_act[31,:])
plt.xlim([0,0.5])

plt.figure()
labels = ['comp1', 'comp2']
vmin = pca_coeff_cuts_act[-1][0,:].mean() - 2 * pca_coeff_cuts_act[-1][0,:].std()
vmax = pca_coeff_cuts_act[-1][0,:].mean() + 2 * pca_coeff_cuts_act[-1][0,:].std()
for t_c in range(len(t_cuts)):
    plt.subplot(2,len(t_cuts),t_c+1)
    plt.title('ExpVar ' + str(np.round(pca_expVar_cuts_act[t_c][0]*100)) + '%')
    mne.viz.plot_topomap(pca_coeff_cuts_act[t_c][0,:], mne.pick_info(A_info_obj_act[2],A_ch_picks_act[2]),vmin=vmin,vmax=vmax)
    
    plt.subplot(2,len(t_cuts),t_c+1 + len(t_cuts))
    plt.title('ExpVar ' + str(np.round(pca_expVar_cuts_act[t_c][1]*100)) + '%')
    mne.viz.plot_topomap(pca_coeff_cuts_act[t_c][1,:], mne.pick_info(A_info_obj_act[2],A_ch_picks_act[2]),vmin=vmin,vmax=vmax)
    
    
#%% Passive Data

data_loc_passive = '/media/ravinderjit/Data_Drive/Data/EEGdata/TemporalCoding/AMmseq_10bits/'
pickle_loc_passive = data_loc_passive + 'Pickles_compActive/'

A_Tot_trials_pass = []
A_Ht_pass = []
A_Htnf_pass = []
A_info_obj_pass = []
A_ch_picks_pass = []

for sub in range(len(Subjects)):
    subject = Subjects[sub]
    with open(os.path.join(pickle_loc_passive,subject +'_AMmseq10bits_compActive.pickle'),'rb') as file:
        [t, Tot_trials, Ht, Htnf, info_obj, ch_picks] = pickle.load(file)
        
    A_Tot_trials_pass.append(Tot_trials)
    A_Ht_pass.append(Ht)
    A_Htnf_pass.append(Htnf)
    A_info_obj_pass.append(info_obj)
    A_ch_picks_pass.append(ch_picks)
    

#%% Get average passive Ht
    
perCh = np.zeros([32,1])
for s in range(len(Subjects)):
    for ch in range(32):
        perCh[ch,0] += np.sum(A_ch_picks_pass[s]==ch)

Avg_Ht_pass = np.zeros([32,t.size])
for s in range(len(Subjects)):
    Avg_Ht_pass[A_ch_picks_pass[s],:] += A_Ht_pass[s]

Avg_Ht_pass = Avg_Ht_pass / perCh
    

#%% PCA on passive t_cuts

pca_sp_cuts_pass = []
pca_expVar_cuts_pass = []
pca_coeff_cuts_pass = []

t_cutT = []
pca = PCA(n_components=2)

for t_c in range(len(t_cuts)):
    if t_c ==0:
        t_1 = np.where(t>=0)[0][0]
    else:
        t_1 = np.where(t>=t_cuts[t_c-1])[0][0]
    
    t_2 = np.where(t>=t_cuts[t_c])[0][0]
    pca_sp = pca.fit_transform(Avg_Ht_pass[:,t_1:t_2].T)
    pca_expVar = pca.explained_variance_ratio_
    pca_coeff = pca.components_
    
    if pca_coeff[0,31] < 0:  # Expand this too look at mutlitple electrodes
       pca_coeff = -pca_coeff
       pca_sp = -pca_sp
       
    
    pca_sp_cuts_pass.append(pca_sp)
    pca_expVar_cuts_pass.append(pca_expVar)
    pca_coeff_cuts_pass.append(pca_coeff)
    t_cutT.append(t[t_1:t_2])

plt.figure()
for t_c in range(len(t_cuts)):
    plt.plot(t_cutT[t_c],pca_sp_cuts_pass[t_c][:,0])
    
plt.figure()
plt.title('2nd component')
for t_c in range(len(t_cuts)):
    plt.plot(t_cutT[t_c],pca_sp_cuts_pass[t_c][:,1])    

plt.figure()
plt.plot(t,Avg_Ht_pass[31,:])
plt.xlim([0,0.5])

plt.figure()
labels = ['comp1', 'comp2']
#vmin = pca_coeff_cuts_pass[-1][0,:].mean() - 2 * pca_coeff_cuts_pass[-1][0,:].std()
#vmax = pca_coeff_cuts_pass[-1][0,:].mean() + 2 * pca_coeff_cuts_pass[-1][0,:].std()
for t_c in range(len(t_cuts)):
    plt.subplot(2,len(t_cuts),t_c+1)
    plt.title('ExpVar ' + str(np.round(pca_expVar_cuts_pass[t_c][0]*100)) + '%')
    mne.viz.plot_topomap(pca_coeff_cuts_pass[t_c][0,:], mne.pick_info(A_info_obj_pass[2],A_ch_picks_pass[2]),vmin=vmin,vmax=vmax)
    
    plt.subplot(2,len(t_cuts),t_c+1 + len(t_cuts))
    plt.title('ExpVar ' + str(np.round(pca_expVar_cuts_act[t_c][1]*100)) + '%')
    mne.viz.plot_topomap(pca_coeff_cuts_pass[t_c][1,:], mne.pick_info(A_info_obj_pass[2],A_ch_picks_pass[2]),vmin=vmin,vmax=vmax)
    
    
#%% Compare Passive Vs Active
    
plt.figure()
plt.plot(t,Avg_Ht_pass[4,:])
plt.plot(t,Avg_Ht_act[4,:])
plt.xlim([0,0.5])
plt.title('Channel Cz')
plt.legend(['Passive','Active'])


plt.figure()
for t_c in range(len(t_cuts)):
    plt.plot(t_cutT[t_c],pca_sp_cuts_pass[t_c][:,0],color='tab:blue')
    plt.plot(t_cutT[t_c],pca_sp_cuts_act[t_c][:,0],color='tab:orange')
plt.legend(['Passive','Active'])

sbp = [4,4]

fig,axs = plt.subplots(sbp[0],sbp[1],sharex=True,gridspec_kw=None)
for p1 in range(sbp[0]):
    for p2 in range(sbp[1]):
        axs[p1,p2].plot(t,Avg_Ht_pass[p1*sbp[1]+p2,:],color='tab:blue')
        axs[p1,p2].plot(t,Avg_Ht_act[p1*sbp[1]+p2,:],color='tab:orange')
        axs[p1,p2].set_title(ch_picks[p1*sbp[1]+p2])    
        axs[p1,p2].set_xlim([0,0.5])
        

fig,axs = plt.subplots(sbp[0],sbp[1],sharex=True,gridspec_kw=None)
for p1 in range(sbp[0]):
    for p2 in range(sbp[1]):
        axs[p1,p2].plot(t,Avg_Ht_pass[p1*sbp[1]+p2+sbp[0]*sbp[1],:],color='tab:blue')
        axs[p1,p2].plot(t,Avg_Ht_act[p1*sbp[1]+p2+sbp[0]*sbp[1],:],color='tab:orange')
        axs[p1,p2].set_title(ch_picks[p1*sbp[1]+p2+sbp[0]*sbp[1]])   
        axs[p1,p2].set_xlim([0,0.5])
        



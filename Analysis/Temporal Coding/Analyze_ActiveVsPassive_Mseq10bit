#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 18:05:45 2021

@author: ravinderjit
"""
import os
import pickle
import numpy as np
import scipy as sp
import mne
import matplotlib.pyplot as plt
from scipy.signal import freqz
from sklearn.decomposition import PCA


pickle_loc = '/media/ravinderjit/Data_Drive/Data/EEGdata/TemporalCoding/AMmseq_bits4/Pickles_full/'

with open(os.path.join(pickle_loc,'S211_'+'AMmseqbits4.pickle'),'rb') as file:
    [tdat, Tot_trials, Ht, Htnf,
     info_obj, ch_picks] = pickle.load(file)

t = tdat[3]
Tot_trials = Tot_trials[3]
Ht = Ht[3]

pickle_loc_3 = '/media/ravinderjit/Data_Drive/Data/EEGdata/TemporalCoding/AMmseq_active/Pickle/'

with open(os.path.join(pickle_loc_3,'S211'+'_AMmseq10bit_Active.pickle'),'rb') as file:
    [t_act, Tot_trials_act, Ht_act,
     info_obj_act, ch_picks_act] = pickle.load(file)
    
sbp = [4,4]
sbp2 = [4,4]
fig,axs =  plt.subplots(sbp[0],sbp[1],sharex=True)

for p1 in range(sbp[0]):
    for p2 in range(sbp[1]):
        cur_ch = p1*sbp[1]+p2
        if np.any(cur_ch==ch_picks):
            ch_ind = np.where(cur_ch==ch_picks)[0][0]
            axs[p1,p2].plot(t,Ht[ch_ind,:])
            axs[p1,p2].set_title(ch_picks[ch_ind])
            axs[p1,p2].set_xlim([0,0.50])
        
        if np.any(cur_ch==ch_picks_act):
            ch_ind = np.where(cur_ch==ch_picks_act)[0][0]
            axs[p1,p2].plot(t,Ht_act[ch_ind,:])
            axs[p1,p2].set_title(ch_picks_act[ch_ind])
            axs[p1,p2].set_xlim([0,0.50])
    
fig.suptitle('Ht ')
fig.legend(['passive','active'])

fig,axs =  plt.subplots(sbp[0],sbp[1],sharex=True)

for p1 in range(sbp[0]):
    for p2 in range(sbp[1]):
        cur_ch = p1*sbp2[1]+p2+sbp[0]*sbp[1]
        if np.any(cur_ch==ch_picks):
            ch_ind = np.where(cur_ch==ch_picks)[0][0]
            axs[p1,p2].plot(t,Ht[ch_ind,:])
            axs[p1,p2].set_title(ch_picks[ch_ind])
            axs[p1,p2].set_xlim([0,0.50])
        
        if np.any(cur_ch==ch_picks_act):
            ch_ind = np.where(cur_ch==ch_picks_act)[0][0]
            axs[p1,p2].plot(t,Ht_act[ch_ind,:])
            axs[p1,p2].set_title(ch_picks_act[ch_ind])
            axs[p1,p2].set_xlim([0,0.50])
    
fig.suptitle('Ht ')
fig.legend(['passive','active'])


#%% PCA

t_cuts = [.015,.040,.120,.500]

pca = PCA(n_components=2)

pca_sp_cuts_pass = [list() for i in range(len(t_cuts))]
pca_expVar_cuts_pass = [list() for i in range(len(t_cuts))]
pca_coeff_cuts_pass = [list() for i in range(len(t_cuts))]

pca_sp_cuts_act = [list() for i in range(len(t_cuts))]
pca_expVar_cuts_act = [list() for i in range(len(t_cuts))]
pca_coeff_cuts_act = [list() for i in range(len(t_cuts))]

A_t_ = [list() for i in range(len(t_cuts))]


for t_c in range(len(t_cuts)):
    if t_c ==0:
        t_1 = np.where(t>=0)[0][0]
    else:
        t_1 = np.where(t>=t_cuts[t_c-1])[0][0]
        
    t_2 = np.where(t>=t_cuts[t_c])[0][0]
    
    pca_sp_cuts_pass[t_c] = pca.fit_transform(Ht[:,t_1:t_2].T)
    pca_expVar_cuts_pass[t_c] = pca.explained_variance_ratio_
    pca_coeff_cuts_pass[t_c] = pca.components_
    
    pca_sp_cuts_act[t_c] = pca.fit_transform(Ht_act[:,t_1:t_2].T)
    pca_expVar_cuts_act[t_c] = pca.explained_variance_ratio_
    pca_coeff_cuts_act[t_c] = pca.components_
    
    if pca_coeff_cuts_pass[t_c][0,-1] < 0:  # Expand this too look at mutlitple electrodes
        pca_coeff_cuts_pass[t_c] = -pca_coeff_cuts_pass[t_c]
        pca_sp_cuts_pass[t_c] = -pca_sp_cuts_pass[t_c]
        
    if pca_coeff_cuts_act[t_c][0,-1] < 0:  # Expand this too look at mutlitple electrodes
        pca_coeff_cuts_act[t_c] = -pca_coeff_cuts_act[t_c]
        pca_sp_cuts_act[t_c] = -pca_sp_cuts_act[t_c]
    
    t_ = t[t_1:t_2]
    A_t_[t_c] = t_


    plt.figure()
    plt.plot(t[t_1:t_2],pca_sp_cuts_pass[t_c][:,0])
    plt.plot(t[t_1:t_2],pca_sp_cuts_act[t_c][:,0])
    plt.title('Comp 1 ' + str(t_cuts[t_c]))
    
    plt.figure()
    plt.plot(t[t_1:t_2],pca_sp_cuts_pass[t_c][:,1])
    plt.plot(t[t_1:t_2],pca_sp_cuts_act[t_c][:,1])
    plt.title('Comp 2 ' + str(t_cuts[t_c]))
    
    
plt.figure()
for t_c in range(len(t_cuts)):
    plt.plot(A_t_[t_c],pca_sp_cuts_pass[t_c][:,0],color='b')
    plt.plot(A_t_[t_c],pca_sp_cuts_act[t_c][:,0],color='r')
    
    
plt.figure()
vmin = pca_coeff_cuts_act[-1][0,:].mean() - 2* pca_coeff_cuts_act[-1][0,:].std()
vmax = pca_coeff_cuts_act[-1][0,:].mean() + 2* pca_coeff_cuts_act[-1][0,:].std()
for t_c in range(len(t_cuts)):
    plt.subplot(2,len(t_cuts),t_c+1)
    plt.title('ExpVar ' + str(np.round(pca_expVar_cuts_pass[t_c][0]*100)) + '%')
    mne.viz.plot_topomap(pca_coeff_cuts_pass[t_c][0,:], mne.pick_info(info_obj,ch_picks),vmin=vmin,vmax=vmax)
    
    plt.subplot(2,len(t_cuts),len(t_cuts)+t_c+1)
    plt.title('ExpVar ' + str(np.round(pca_expVar_cuts_act[t_c][0]*100)) + '%')
    mne.viz.plot_topomap(pca_coeff_cuts_act[t_c][0,:], mne.pick_info(info_obj_act,ch_picks_act),vmin=vmin,vmax=vmax)
    


























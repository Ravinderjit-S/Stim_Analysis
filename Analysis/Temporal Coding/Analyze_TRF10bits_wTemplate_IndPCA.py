#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 16:48:24 2021

@author: ravinderjit
"""

import os
import pickle
import numpy as np
import mne
import matplotlib.pyplot as plt
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


#%% Load Template
template_loc = '/media/ravinderjit/Data_Drive/Data/EEGdata/TemporalCoding/AMmseq_10bits/Pickles/PCA_passive_template.pickle'

with open(template_loc,'rb') as file:
    [pca_coeffs_cuts,pca_expVar_cuts,t_cuts] = pickle.load(file)
    
bstemTemplate  = pca_coeffs_cuts[0][0,:]
cortexTemplate = pca_coeffs_cuts[2][0,:]


#%% Load Passive Data


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

print('Done Loading Passive data')

#%% Load Active Counting Data
    
A_Tot_trials_count = []
A_Ht_count = []
A_Htnf_count = []
A_info_obj_count = []
A_ch_picks_count = []

for sub in range(len(Subjects)):
    subject = Subjects[sub]
    with open(os.path.join(pickle_loc,subject +'_AMmseq10bits_Active.pickle'),'rb') as file:
        [t, Tot_trials, Ht, Htnf, info_obj, ch_picks] = pickle.load(file)
        
    A_Tot_trials_count.append(Tot_trials)
    A_Ht_count.append(Ht)
    A_Htnf_count.append(Htnf)
    A_info_obj_count.append(info_obj)
    A_ch_picks_count.append(ch_picks)
    
print('Done Loading Counting data')
#%% Load active shift detect task
    
Subjects_sd = ['S207', 'S228', 'S236', 'S238', 'S239'] #Leaving out S211 for now

pickle_loc_sd = '/media/ravinderjit/Data_Drive/Data/EEGdata/TemporalCoding/AMmseq_active_harder/Pickles/'
A_Tot_trials_sd = []
A_Ht_sd = []
A_info_obj_sd = []
A_ch_picks_sd = []

for sub in range(len(Subjects_sd)):
    subject = Subjects_sd[sub]
    with open(os.path.join(pickle_loc_sd,subject +'_AMmseq10bit_Active_harder.pickle'),'rb') as file:
        [t, Tot_trials, Ht, info_obj, ch_picks] = pickle.load(file)
    
    A_Tot_trials_sd.append(Tot_trials)
    A_Ht_sd.append(Ht)
    A_info_obj_sd.append(info_obj)
    A_ch_picks_sd.append(ch_picks)

print('Done Loading Shift Detect data')
#%% Plot Ch. Cz

for sub in range(len(Subjects)):
    plt.figure()
    ch_pass_ind = np.where(A_ch_picks_pass[sub] == 31)[0][0]
    ch_pass_count = np.where(A_ch_picks_count[sub] == 31)[0][0]
    plt.plot(t,A_Ht_pass[sub][ch_pass_ind,:],label='Passive')
    plt.plot(t,A_Ht_count[sub][ch_pass_count,:],label='Count')
    if (subject in Subjects_sd):
        sub_sd = Subjects_sd.index(subject)
        ch_pass_sd = np.where(A_ch_picks_sd[sub_sd] ==31)[0][0]
        plt.plot(t,A_Ht_sd[sub_sd][ch_pass_sd,:],label='Shift Detect')
    plt.xlim([0,0.5])
    plt.title(Subjects[sub])
    plt.legend()

#%% Extract responses from templates

t_var_step = .010
t_var = np.arange(t_var_step,0.5+t_var_step,t_var_step)

for sub in range(len(Subjects)):
    bstemResp_pass = np.matmul(A_Ht_pass[sub].T,bstemTemplate[A_ch_picks_pass[sub]])
    cortexResp_pass = np.matmul(A_Ht_pass[sub].T,cortexTemplate[A_ch_picks_pass[sub]])
    
    bstem_expVar = np.zeros(t_var.shape)
    cortex_expVar = np.zeros(t_var.shape)    
    for tt in range(t_var.size):
        tt1 = np.where(t>=t_var[tt] - t_var_step)[0][0]
        tt2 = np.where(t>=t_var[tt])[0][0]
        
        true_resp = A_Ht_pass[sub][:,tt1:tt2]

        bstem_est = np.matmul(bstemTemplate[A_ch_picks_pass[sub],np.newaxis],bstemResp_pass[np.newaxis,tt1:tt2])
        # bstem_est = (bstem_est-bstem_est.mean(axis=1)[:,np.newaxis]) #demean
        # bstem_est = bstem_est * (true_resp.var(axis=1) /bstem_est.var(axis=1))[:,np.newaxis] #equalize variance
        bstem_expVar[tt] = explained_variance_score(true_resp, bstem_est, multioutput='variance_weighted')  
        
        cortex_est = np.matmul(cortexTemplate[A_ch_picks_pass[sub],np.newaxis],cortexResp_pass[np.newaxis,tt1:tt2])
        cortex_expVar[tt] = explained_variance_score(true_resp, cortex_est, multioutput='variance_weighted')    
    
    fig,ax = plt.subplots(2,1)
    ax[0].plot(t,bstemResp_pass)
    ax[0].set_xlabel('Time (sec)')
    ax[0].set_title(Subjects[sub] + ' Brainstem')
    ax[0].set_xlim([0,0.5])
    ax0 = ax[0].twinx()
    ax0.plot(t_var,bstem_expVar,color='tab:orange')
    ax0.plot(t_var,cortex_expVar,color='green')
    ax0.set_ylim([0,1])
    
    
    ax[1].plot(t,cortexResp_pass)
    ax[1].set_xlabel('Time (sec)')
    ax[1].set_title('Cortex')
    ax[1].set_xlim([0,0.5])
    ax1 = ax[1].twinx()
    ax1.plot(t_var,cortex_expVar, color='tab:orange')
    ax1.set_ylim([0,1])

#%% Individual PCA analysis








    
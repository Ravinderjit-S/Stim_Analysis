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

plt.figure()
plt.title('Componenet 2')
for t_c in range(len(t_cuts)):
    plt.plot(t_cutT[t_c],pca_sp_cuts_pass[t_c][:,1],color='tab:blue')
    plt.plot(t_cutT[t_c],pca_sp_cuts_act[t_c][:,1],color='tab:orange')
plt.legend(['Passive','Active'])

plt.figure()
plt.title('Active second peak')
plt.plot(t_cutT[1],pca_sp_cuts_act[1][:,0],color='tab:blue')
plt.plot(t_cutT[1],pca_sp_cuts_act[1][:,1],color='tab:orange')
plt.xlabel('Time (sec)')
plt.legend(['Comp1', 'Comp2'])

plt.figure()
plt.title('Passive second peak')
plt.plot(t_cutT[1],pca_sp_cuts_pass[1][:,0],color='tab:blue')
plt.plot(t_cutT[1], - pca_sp_cuts_pass[1][:,1],color='tab:orange')
plt.xlabel('Time (sec)')
plt.legend(['Comp1', '- Comp2'])


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
        

for t_c in range(len(t_cuts)):
    plt.figure()
    plt.plot(pca_coeff_cuts_pass[t_c][0,:],color='tab:blue')
    plt.plot(pca_coeff_cuts_act[t_c][0,:],color='tab:orange')
    
    
#%% Individual subject Passive Vs Active
    
pca_sp_cuts_sub_act = []
pca_expVar_cuts_sub_act = []

pca_sp_cuts_sub_pass = []
pca_expVar_cuts_sub_pass = []

for sub in range(len(Subjects)):
    pca_sp_cuts_act_ = []
    pca_expVar_cuts_act_ = []
    
    pca_sp_cuts_pass_ = []
    pca_expVar_cuts_pass_ = []
    
    H_t_act = A_Ht_act[sub].T
    H_t_pass = A_Ht_pass[sub].T    
    
    for t_c in range(len(t_cuts)):
        if t_c ==0:
            t_1 = np.where(t>=0)[0][0]
        else:
            t_1 = np.where(t>=t_cuts[t_c-1])[0][0]
        
        t_2 = np.where(t>=t_cuts[t_c])[0][0]
        
        H_tc_act = H_t_act[t_1:t_2,:] - H_t_act[t_1:t_2,:].mean(axis=0)[np.newaxis,:]
        H_tc_pass = H_t_pass[t_1:t_2,:] - H_t_pass[t_1:t_2,:].mean(axis=0)[np.newaxis,:]
        
        pca_sp_act = np.matmul(H_tc_act,pca_coeff_cuts_act[t_c][0,A_ch_picks_act[sub]])
        pca_sp_pass = np.matmul(H_tc_pass,pca_coeff_cuts_pass[t_c][0,A_ch_picks_pass[sub]])
        
        H_tc_act_est = np.matmul(pca_coeff_cuts_act[t_c][0,A_ch_picks_act[sub]][:,np.newaxis],pca_sp_act[np.newaxis,:])
        H_tc_pass_est = np.matmul(pca_coeff_cuts_pass[t_c][0,A_ch_picks_pass[sub]][:,np.newaxis],pca_sp_pass[np.newaxis,:])
        
        pca_expVar_act = explained_variance_score(H_tc_act,H_tc_act_est.T,multioutput='variance_weighted')
        pca_expVar_pass = explained_variance_score(H_tc_pass,H_tc_pass_est.T,multioutput='variance_weighted')
        
        pca_sp_cuts_act_.append(pca_sp_act)
        pca_sp_cuts_pass_.append(pca_sp_pass)
        
        pca_expVar_cuts_act_.append(pca_expVar_act)
        pca_expVar_cuts_pass_.append(pca_expVar_pass)
        
    pca_sp_cuts_sub_act.append(pca_sp_cuts_act_)
    pca_sp_cuts_sub_pass.append(pca_sp_cuts_pass_)
    
    pca_expVar_cuts_sub_act.append(pca_expVar_cuts_act_)
    pca_expVar_cuts_sub_pass.append(pca_expVar_cuts_pass_)
    
    
for sub in range(len(Subjects)):
    plt.figure()
    plt.title(Subjects[sub])
    for t_c in range(len(t_cuts)):
        plt.plot(t_cutT[t_c],pca_sp_cuts_sub_pass[sub][t_c],color='tab:blue')
        plt.plot(t_cutT[t_c],pca_sp_cuts_sub_act[sub][t_c],color='tab:orange')
        
for sub in range(len(Subjects)):
    plt.figure()
    plt.title('Ch.Cz: ' + Subjects[sub])
    ch_ind_pass = np.where(A_ch_picks_pass[sub] == 15)[0][0]
    ch_ind_act = np.where(A_ch_picks_act[sub] == 15)[0][0]
    plt.plot(t,A_Ht_pass[sub][ch_ind_pass,:])
    plt.plot(t, A_Ht_act[sub][ch_ind_act,:])
    plt.xlim([0,0.5])
    
        
        
#%% Individual Subject, Individual channel
sbp = [4,4]
sbp2 = sbp

for sub in range(len(Subjects)):
    fix,axs = plt.subplots(sbp[0],sbp[1],sharex=True)
    Ht_act_ = A_Ht_act[sub]
    Ht_pass_ = A_Ht_pass[sub]
    
    ch_picks_act_ = A_ch_picks_act[sub]
    ch_picks_pass_ = A_ch_picks_pass[sub]
    
    for p1 in range(sbp[0]):
        for p2 in range(sbp[1]):
            cur_ch = p1*sbp[1] + p2
            if np.any(cur_ch ==ch_picks_pass_):
                ch_ind = np.where(cur_ch == ch_picks_pass_)[0][0]
                axs[p1,p2].plot(t,Ht_pass_[ch_ind,:],color='tab:blue')
                axs[p1,p2].set_title(ch_picks_pass_[ch_ind])
                axs[p1,p2].set_xlim([-0.050,0.5])
            
            if np.any(cur_ch == ch_picks_act_):
                ch_ind = np.where(cur_ch == ch_picks_act_)[0][0]
                axs[p1,p2].plot(t,Ht_act_[ch_ind,:],color='tab:orange')
                axs[p1,p2].set_title(ch_picks_act_[ch_ind])
                axs[p1,p2].set_xlim([-0.050,0.5])
    
    plt.title(Subjects[sub])      


for sub in range(len(Subjects)):
    fix,axs = plt.subplots(sbp[0],sbp[1],sharex=True)
    Ht_act_ = A_Ht_act[sub]
    Ht_pass_ = A_Ht_pass[sub]
    
    ch_picks_act_ = A_ch_picks_act[sub]
    ch_picks_pass_ = A_ch_picks_pass[sub]
    
    for p1 in range(sbp[0]):
        for p2 in range(sbp[1]):
            cur_ch = p1*sbp2[1]+p2+sbp[0]*sbp[1]
            if np.any(cur_ch ==ch_picks_pass_):
                ch_ind = np.where(cur_ch == ch_picks_pass_)[0][0]
                axs[p1,p2].plot(t,Ht_pass_[ch_ind,:],color='tab:blue')
                axs[p1,p2].set_title(ch_picks_pass_[ch_ind])
                axs[p1,p2].set_xlim([-0.050,0.5])
            
            if np.any(cur_ch == ch_picks_act_):
                ch_ind = np.where(cur_ch == ch_picks_act_)[0][0]
                axs[p1,p2].plot(t,Ht_act_[ch_ind,:],color='tab:orange')
                axs[p1,p2].set_title(ch_picks_act_[ch_ind])
                axs[p1,p2].set_xlim([-0.050,0.5])
    
    plt.title(Subjects[sub])      


#%% Compare passive vs Active 



#%% Active shift detect task
    
Subjects_sd = ['S207', 'S211', 'S228', 'S236', 'S238', 'S239']

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


#Get average Ht
perCh = np.zeros([32,1])
for s in range(len(Subjects_sd)):
    for ch in range(32):
        perCh[ch,0]+= np.sum(A_ch_picks_sd[s]==ch)

Avg_Ht_sd = np.zeros([32,t.size])
for s in range(len(Subjects_sd)):
    Avg_Ht_sd[A_ch_picks_sd[s],:] += A_Ht_sd[s]
    
Avg_Ht_sd = Avg_Ht_sd / perCh

#PCA on t_cuts

pca_sp_cuts_sd = []
pca_expVar_cuts_sd = []
pca_coeff_cuts_sd = []

t_cutT = []
pca = PCA(n_components=2)

for t_c in range(len(t_cuts)):
    if t_c ==0:
        t_1 = np.where(t>=0)[0][0]
    else:
        t_1 = np.where(t>=t_cuts[t_c-1])[0][0]
        
    t_2 = np.where(t>=t_cuts[t_c])[0][0]
    
    pca_sp = pca.fit_transform(Avg_Ht_sd[:,t_1:t_2].T)
    pca_expVar = pca.explained_variance_ratio_
    pca_coeff = pca.components_
    
    if pca_coeff[0,31] < 0:  # Expand this too look at mutlitple electrodes
       pca_coeff = -pca_coeff
       pca_sp = -pca_sp
       
    pca_sp_cuts_sd.append(pca_sp)
    pca_expVar_cuts_sd.append(pca_expVar)
    pca_coeff_cuts_sd.append(pca_coeff)
    t_cutT.append(t[t_1:t_2])

plt.figure()
for t_c in range(len(t_cuts)):
    plt.plot(t_cutT[t_c],pca_sp_cuts_act[t_c][:,0])

plt.figure()
plt.plot(t,Avg_Ht_sd[31,:],label='Shift Detect')
plt.plot(t,Avg_Ht_act[31,:], label='Counting')
plt.plot(t,Avg_Ht_pass[31,:],label='passive')
plt.legend()
plt.xlim([0,0.5])

plt.figure()
labels = ['comp1', 'comp2']
vmin = pca_coeff_cuts_sd[-1][0,:].mean() - 2 * pca_coeff_cuts_sd[-1][0,:].std()
vmax = pca_coeff_cuts_sd[-1][0,:].mean() + 2 * pca_coeff_cuts_sd[-1][0,:].std()
for t_c in range(len(t_cuts)):
    plt.subplot(2,len(t_cuts),t_c+1)
    plt.title('ExpVar ' + str(np.round(pca_expVar_cuts_sd[t_c][0]*100)) + '%')
    mne.viz.plot_topomap(pca_coeff_cuts_sd[t_c][0,:], mne.pick_info(A_info_obj_sd[1],A_ch_picks_sd[1]),vmin=vmin,vmax=vmax)
    
    plt.subplot(2,len(t_cuts),t_c+1 + len(t_cuts))
    plt.title('ExpVar ' + str(np.round(pca_expVar_cuts_sd[t_c][1]*100)) + '%')
    mne.viz.plot_topomap(pca_coeff_cuts_sd[t_c][1,:], mne.pick_info(A_info_obj_sd[1],A_ch_picks_sd[1]),vmin=vmin,vmax=vmax)
    
A_Ht_sd
#%% Active Shift Detect individual subject
    
plt.figure()
sub_sd_ind = 5
sub_ind = Subjects.index(Subjects_sd[sub_sd_ind])
ch = 31
ch_ind_sd = np.where(A_ch_picks_sd[sub_sd_ind]==ch)[0][0]
ch_ind_pass =  np.where(A_ch_picks_pass[sub_ind]==ch)[0][0]
ch_ind_act = np.where(A_ch_picks_act[sub_ind]==ch)[0][0]
plt.plot(t,A_Ht_pass[sub_ind][ch_ind_pass,:],label='Passive')
plt.plot(t,A_Ht_act[sub_ind][ch_ind_act,:], label='Active')
plt.plot(t,A_Ht_sd[sub_sd_ind][ch_ind_sd,:], label='Shift Detect')
plt.xlim([0,0.5])
plt.title(Subjects_sd[sub_sd_ind])
plt.legend()

    

















    

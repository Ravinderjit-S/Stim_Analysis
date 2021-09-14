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
from scipy.signal import find_peaks

import sys
sys.path.append(os.path.abspath('../ACRanalysis/'))
from ACR_helperFuncs import PCA_tcuts

def PCA_tcutsPlots(t_cuts,pca_spCuts,comp,t,cz, title_):
    plt.figure()
    for t_c in range(len(t_cuts)):
        plt.plot(t_cuts[t_c],pca_spCuts[t_c][:,comp])
    plt.plot(t,cz,color='k')
    plt.xlim([0,0.3])
    plt.title(title_)
    
def PCA_tcuts_topomap(pca_coeffCuts, t_cuts, pca_expVarCuts, infoObj, ch_picks, title_):
    plt.figure()
    plt.suptitle(title_)
    vmin = pca_coeffCuts[-1][0,:].mean() - 2 * pca_coeffCuts[-1][0,:].std()
    vmax = pca_coeffCuts[-1][0,:].mean() + 2 * pca_coeffCuts[-1][0,:].std()
    for t_c in range(len(t_cuts)):
        plt.subplot(2,len(t_cuts),t_c+1)
        plt.title('ExpVar ' + str(np.round(pca_expVarCuts[t_c][0]*100)) + '%')
        mne.viz.plot_topomap(pca_coeffCuts[t_c][0,:], mne.pick_info(infoObj,ch_picks),vmin=vmin,vmax=vmax)
        
        plt.subplot(2,len(t_cuts),t_c+1 + len(t_cuts))
        plt.title('ExpVar ' + str(np.round(pca_expVarCuts[t_c][1]*100)) + '%')
        mne.viz.plot_topomap(pca_coeffCuts[t_c][1,:], mne.pick_info(infoObj,ch_picks),vmin=vmin,vmax=vmax)
        
        
    
        



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
    
    cz_pass = A_Ht_pass[sub][ch_pass_ind,:]
    cz_count = A_Ht_count[sub][ch_pass_count,:]
    
    [peaks_pass,properties] = find_peaks(-cz_pass)
    [peaks_count,properties] = find_peaks(-cz_count)
    
    [pk_pass,properties] = find_peaks(cz_pass)
    [pk_count,properties] = find_peaks(cz_count)
    
    plt.plot(t,cz_pass,label='Passive')
    plt.scatter(t[peaks_pass],cz_pass[peaks_pass],color='blue')
    plt.scatter(t[pk_pass],cz_pass[pk_pass],color='blue')
    
    plt.plot(t,cz_count,label='Count')
    plt.scatter(t[peaks_count],cz_count[peaks_count],color='red')
    plt.scatter(t[pk_count],cz_count[pk_count],color='red')
    
    
    if (subject in Subjects_sd):
        sub_sd = Subjects_sd.index(subject)
        ch_pass_sd = np.where(A_ch_picks_sd[sub_sd] ==31)[0][0]
        cz_sd = A_Ht_sd[sub_sd][ch_pass_sd,:]
        
        [peaks_sd, properties] = find_peaks(-cz_sd)
        [pk_sd, properties] = find_peaks(cz_sd)
        
        plt.plot(t,cz_sd,label='Shift Detect')
        plt.scatter(t[peaks_sd],cz_sd[peaks_sd],color='black')
        plt.scatter(t[pk_sd],cz_sd[pk_sd],color='black')
        
    plt.xlim([0,0.5])
    plt.title(Subjects[sub])
    plt.legend()

    
#%% Manual Peak and cutoff reading
    
cuts_passive = []
cuts_count = []
cuts_sd = []

#S207
cuts_passive.append([.020, .035, .068, .134, .282 ])
cuts_count.append([.018, .034, .068,  .130, .262])
cuts_sd.append([.0124, .034, .063, .118, .237])

#S228
cuts_passive.append([.016, .047, .069, .127, .258 ])
cuts_count.append([.014, .040, .070, .127, .242 ])
cuts_sd.append([.012, .034, .063, .118, .237  ])

#Peak Latencies
pk_passive = []
pk_count = []
pk_sd = []

#S207
pk_passive.append([.007, .028, .047, .092, .201])
pk_count.append([.007, .027, .046, .094, .186 ])
pk_sd.append([.004, .022, .043, .097, .169])

#228
pk_passive.append([.0073, .037, .058, .091, .189])
pk_count.append([.0069, .023, .058, .091, .170 ])
pk_sd.append([.0044, .022, .043, .097, .169])

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
Sub_ind = 1

#Passive PCA t cuts
if(Sub_ind==0):
    remove_ch_pass = [np.where(A_ch_picks_pass[Sub_ind]==13)[0][0], 
                 np.where(A_ch_picks_pass[Sub_ind]==15)[0][0]]
elif(Sub_ind==1):
    remove_ch_pass = [np.where(A_ch_picks_pass[Sub_ind]==14)[0][0],
        np.where(A_ch_picks_pass[Sub_ind]==15)[0][0],
        np.where(A_ch_picks_pass[Sub_ind]==16)[0][0],
         np.where(A_ch_picks_pass[Sub_ind]==27)[0][0]
        ]
    


Ht_pass = A_Ht_pass[Sub_ind] 
t_cuts_pass = cuts_passive[Sub_ind]
ch_picks_pass = A_ch_picks_pass[Sub_ind]
chs_use_pass = np.arange(A_ch_picks_pass[Sub_ind].size)
chs_use_pass = np.delete(chs_use_pass,remove_ch_pass)

[pca_spCuts_pass, pca_expVarCuts_pass, pca_coeffCuts_pass, t_cuts_pass] = PCA_tcuts(
    Ht_pass, t, t_cuts_pass, ch_picks_pass, chs_use_pass)

#Count PCA t cuts
remove_ch_count = [np.where(A_ch_picks_count[Sub_ind]==13)[0][0], 
             np.where(A_ch_picks_count[Sub_ind]==15)[0][0],
             np.where(A_ch_picks_count[Sub_ind]==10)[0][0],
             ]

Ht_count = A_Ht_count[Sub_ind] 
t_cuts_count = cuts_count[Sub_ind]
ch_picks_count = A_ch_picks_count[Sub_ind]
chs_use_count = np.arange(A_ch_picks_count[Sub_ind].size)
chs_use_count = np.delete(chs_use_count,remove_ch_count)

[pca_spCuts_count, pca_expVarCuts_count, pca_coeffCuts_count, t_cuts_count] = PCA_tcuts(
    Ht_count, t, t_cuts_count, ch_picks_count, chs_use_count)

#SD PCA t cuts
remove_ch_sd = [np.where(A_ch_picks_sd[Sub_ind]==13)[0][0], 
             np.where(A_ch_picks_sd[Sub_ind]==15)[0][0],
              np.where(A_ch_picks_sd[Sub_ind]==10)[0][0],
             ]

Ht_sd = A_Ht_sd[Sub_ind]
t_cuts_sd = cuts_sd[Sub_ind]
ch_picks_sd = A_ch_picks_sd[Sub_ind]
chs_use_sd = np.arange(A_ch_picks_sd[Sub_ind].size)
chs_use_sd = np.delete(chs_use_sd,remove_ch_sd) 

[pca_spCuts_sd, pca_expVarCuts_sd, pca_coeffCuts_sd, t_cuts_sd] = PCA_tcuts(
    Ht_sd, t, t_cuts_sd, ch_picks_sd, chs_use_sd)

#Full PCA passive
t_1 = np.where(t>=0)[0][0]
t_2 = np.where(t>=0.3)[0][0]

pca = PCA(n_components=2)
t_fullPCA = t[t_1:t_2]
pca_sp_full = pca.fit_transform(Ht_pass[:,t_1:t_2].T)
pca_expVar_full = pca.explained_variance_ratio_
pca_coeff_full = pca.components_

if pca_coeff_full[0,A_ch_picks_pass[Sub_ind]==31] < 0:  #Consider to Expand this too look at mutlitple electrodes
   pca_coeff_full = -pca_coeff_full
   pca_sp_full = -pca_sp_full

#%% Plot PCA tcuts
   
# Plot Passive
ch_pass_ind = np.where(A_ch_picks_pass[Sub_ind] == 31)[0][0]
cz_pass = A_Ht_pass[Sub_ind][ch_pass_ind,:]


PCA_tcutsPlots(t_cuts_pass,pca_spCuts_pass,0,t,cz_pass, 'Passive: ' + Subjects[Sub_ind])
PCA_tcutsPlots(t_cuts_pass,pca_spCuts_pass,1,t,cz_pass, 'Passive Comp 2: ' + Subjects[Sub_ind])
PCA_tcuts_topomap(pca_coeffCuts_pass, t_cuts_pass, pca_expVarCuts_pass, A_info_obj_pass[Sub_ind], 
                  A_ch_picks_pass[Sub_ind][chs_use_pass], 'Passive: ' + Subjects[Sub_ind])

# Plot count
ch_pass_count = np.where(A_ch_picks_count[Sub_ind] == 31)[0][0]
cz_count = A_Ht_count[Sub_ind][ch_pass_count,:]


PCA_tcutsPlots(t_cuts_count,pca_spCuts_count,0,t,cz_count, 'Count: ' + Subjects[Sub_ind])
PCA_tcutsPlots(t_cuts_count,pca_spCuts_count,1,t,cz_count, 'Count Comp 2: ' + Subjects[Sub_ind])
PCA_tcuts_topomap(pca_coeffCuts_count, t_cuts_count, pca_expVarCuts_count, A_info_obj_count[Sub_ind], 
                  A_ch_picks_count[Sub_ind][chs_use_count], 'Count: ' + Subjects[Sub_ind])
    
# Plot SD
ch_pass_sd = np.where(A_ch_picks_sd[Sub_ind] == 31)[0][0]
cz_sd = A_Ht_sd[Sub_ind][ch_pass_sd,:]

PCA_tcutsPlots(t_cuts_sd,pca_spCuts_sd,0,t,cz_sd, 'SD: ' + Subjects[Sub_ind])
PCA_tcutsPlots(t_cuts_sd,pca_spCuts_sd,1,t,cz_sd, 'SD Comp 2: ' + Subjects[Sub_ind])
PCA_tcuts_topomap(pca_coeffCuts_sd, t_cuts_sd, pca_expVarCuts_sd, A_info_obj_sd[Sub_ind], 
                  A_ch_picks_sd[Sub_ind][chs_use_sd], 'Count: ' + Subjects[Sub_ind])

#%% Plot 32 channel with tsplit labelled









    
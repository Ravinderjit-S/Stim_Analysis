#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 12:02:23 2020

@author: ravinderjit
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat 
import matplotlib.colors as mcolors
import pickle

Mod = [131, 139]

StimData_2_10 = loadmat('/home/ravinderjit/Documents/OnlineStim_WavFiles/CMR_frozenMod/Mod_' + str(Mod[0]) +'_'+str(Mod[1]) +'/Stim_Data.mat')
correct = StimData_2_10['correct'].squeeze()
SNRdB_exp = StimData_2_10['SNRdB_exp'].squeeze()
SNRdB = SNRdB_exp[:,0]
coh = SNRdB_exp[:,1]

#Results_fname = ['CMR3AFC_2_10_Rav_results.json']
#Results_fname = ['CMR3AFC_2_10_frozen_Rav_results.json']
#Results_fname = ['CMR3AFC_36_44_Rav_results.json']
#Results_fname = ['CMR3AFC_36_44_frozen_Rav_results.json']
Results_fname = ['CMR3AFC_131_139_Rav_results.json']

with open(Results_fname[0]) as f:
    results = json.load(f)
    
trialnum = np.zeros((len(correct),len(results)))
SNR_ = np.zeros((len(correct),len(results)))
coh_ = np.zeros((len(correct),len(results)))
resps = np.zeros((len(correct),len(results)))
corr_check = np.zeros((len(correct),len(results)))
subjects = []

for k in range(0,len(results)):
    subjects.append(results[k][0]['subject'])
    subj = results[k]
    cur_t = 0
    for trial in subj:
        if not 'annot' in trial:
            continue
        if len(trial['annot']) == 0:
            continue
        
        trialnum[cur_t,k] = trial['trialnum']
        annot = trial['annot']
        SNR_[cur_t,k] = int(annot[annot.find(':')+1:annot.find('}')])
        coh_[cur_t,k] = trial['cond']
        resps[cur_t,k] = int(trial['button_pressed'])
        corr_check[cur_t,k] = trial['correct']
        cur_t+=1
        
resps = resps+1
coh_ = coh_ -1

#SNRs_1 = [4,-2,-8,-14,-20,-26,-32,-38]
SNRs_2 = [6,0,-6,-12,-18,-24,-30]
SNRs_1 = SNRs_2

SNRs_1_acc = np.zeros((len(SNRs_1),len(subjects)))
for j in range(0,len(SNRs_1)):
    mask = (SNR_==SNRs_1[j]) & (coh_==1)
    SNRs_1_acc[j,:] = corr_check[mask[:,0],:].sum(axis=0) / corr_check[mask[:,0]].shape[0]
    
SNRs_2_acc = np.zeros((len(SNRs_2),len(subjects)))
for j in range(0,len(SNRs_2)):
    mask = (SNR_==SNRs_2[j]) & (coh_==0)
    SNRs_2_acc[j] = corr_check[mask[:,0],:].sum(axis=0)/corr_check[mask[:,0]].shape[0]
    
# if Mod == [36,44]: #remove demo subject
#     SNRs_1_acc = SNRs_1_acc[:,1:] 
#     SNRs_2_acc = SNRs_2_acc[:,1:] 
    
plt.figure()
plt.plot(SNRs_1, SNRs_1_acc,color='blue',label='coh')
plt.plot(SNRs_2, SNRs_2_acc,color='red',label='incoh')
plt.legend()
#plt.title('frozen')
    












#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 21:38:51 2020

@author: ravinderjit
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat 
import matplotlib.colors as mcolors

StimData = ['../../../Stimuli/TemporalCoding/OnlineExp_AMphi/StimData_4.mat']
StimData.append('../../../Stimuli/TemporalCoding/OnlineExp_AMphi/StimData_8.mat')
StimData.append('../../../Stimuli/TemporalCoding/OnlineExp_AMphi/StimData_16.mat')
StimData.append('../../../Stimuli/TemporalCoding/OnlineExp_AMphi/StimData_32.mat')
StimData.append('../../../Stimuli/TemporalCoding/OnlineExp_AMphi/StimData_64.mat')
StimData.append('../../../Stimuli/TemporalCoding/OnlineExp_AMphi/StimData_128.mat')

Results_fname = ['Task_AMphi_AM4_Rav_results.json']
Results_fname.append('Task_AMphi_AM8_Rav_results.json')
Results_fname.append('Task_AMphi_AM16_Rav_results.json')
Results_fname.append('Task_AMphi_AM32_Rav_results.json')
Results_fname.append('Task_AMphi_AM64_Rav_results.json')
Results_fname.append('Task_AMphi_AM128_Rav_results.json')

AM = [4,8,16,32,64,128]

AM_avgs = np.zeros((4,len(AM)))

for j in range(0,len(StimData)):
    StimData_j = StimData[j]
    Results_fname_j = Results_fname[j]
    
    stim_info = loadmat(StimData_j)
    correct = stim_info['correct'].squeeze()
    stim_phis = stim_info['phis'].squeeze()
    correct = correct -1
    
    with open(Results_fname_j) as f:
        results = json.load(f)
    subjects = []
    trialnum = np.zeros((len(correct),len(results)))
    phi = np.zeros((len(correct),len(results)))
    resps = np.zeros((len(correct),len(results)))
    rt = np.zeros((len(correct),len(results)))
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
            phi[cur_t,k] = int(annot[annot.find(':')+1:annot.find('}')])
            resps[cur_t,k] = int(trial['button_pressed'])
            rt[cur_t,k] = trial['rt']
            cur_t +=1 
        


    resps = resps+1

    phi_conds = np.unique(phi)
    phi = phi[:,0]
    accuracy_conds = np.zeros((len(phi_conds), len(subjects)))
    for ll in range(len(phi_conds)):
        mask = phi==phi_conds[ll]
        for m in range(0,len(subjects)):
            accuracy = np.sum(resps[mask,m] == correct[mask]) / len(correct[mask])
            accuracy_conds[ll,m] = accuracy
        
    
    fig, ax = plt.subplots()
    ax.plot(range(len(phi_conds)),accuracy_conds)
    plt.xticks(range(len(phi_conds)),labels = phi_conds)
    plt.ylabel('Accuracy')
    plt.xlabel('Phase Difference')
    plt.title(str(AM[j]))
    #plt.legend(subjects)
    
    sem = accuracy_conds.std(axis=1) / np.sqrt(accuracy_conds.shape[1])
    
    fix,ax = plt.subplots()
    ax.plot(range(len(phi_conds)),accuracy_conds,color=mcolors.CSS4_COLORS['silver'])
    ax.errorbar(range(len(phi_conds)),accuracy_conds.mean(axis=1),yerr = sem,color='black',linewidth=2)
    plt.xticks(range(len(phi_conds)),labels = phi_conds)
    plt.ylabel('Accuracy')
    plt.xlabel('Phase Difference')
    plt.title(str(AM[j]))
    
    AM_avgs[:,j] = accuracy_conds.mean(axis=1)
    
fig, ax = plt.subplots()
ax.plot(range(len(phi_conds)),AM_avgs)
plt.xticks(range(len(phi_conds)),labels = phi_conds)
plt.ylabel('Accuracy')
plt.xlabel('Phase Difference')
plt.legend(AM)

t_diff = np.zeros(AM_avgs.shape)
for m in range (len(AM)):
    t_diff[:,m] = (1/AM[m]) * phi_conds.T/360


fig, ax = plt.subplots()
ax.plot(t_diff,AM_avgs,marker='x',linewidth=0)
plt.ylabel('Accuracy')
plt.xlabel('Time Difference')
plt.legend(AM)




    

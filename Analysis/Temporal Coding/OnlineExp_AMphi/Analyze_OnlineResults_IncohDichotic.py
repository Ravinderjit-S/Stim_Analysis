#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 21:38:51 2020

@author: ravinderjit
"""

import json
import numpy as np
import scipy.stats as spst
import matplotlib.pyplot as plt
from scipy.io import loadmat 
import matplotlib.colors as mcolors
import pickle

StimData = ['../../../Stimuli/TemporalCoding/OnlineExp_AMIncohphi/StimData_diotic4.mat']
StimData.append('../../../Stimuli/TemporalCoding/OnlineExp_AMIncohphi/StimData_diotic8.mat')
StimData.append('../../../Stimuli/TemporalCoding/OnlineExp_AMIncohphi/StimData_diotic16.mat')
StimData.append('../../../Stimuli/TemporalCoding/OnlineExp_AMIncohphi/StimData_diotic32.mat')
StimData.append('../../../Stimuli/TemporalCoding/OnlineExp_AMIncohphi/StimData_diotic64.mat')

Results_fname = ['Task_AMphi_Incoh_4_Rav_results.json']
Results_fname.append('Task_AMphi_Incoh_8_Rav_results.json')
Results_fname.append('Task_AMphi_Incoh_16_Rav_results.json')
Results_fname.append('Task_AMphi_Incoh_32_Rav_results.json')
Results_fname.append('Task_AMphi_Incoh_64_Rav_results.json')

AM = [4,8,16,32,64]

AM_med = np.zeros((3,len(AM)))
AM_mad_sem = np.zeros((3,len(AM)))
AM_avgs = np.zeros((3,len(AM)))
AM_sems = np.zeros((3,len(AM)))
AM_rav = np.zeros((3,len(AM)))

for j in range(0,len(AM)):
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
    plt.legend(subjects)
    
    sem = accuracy_conds.std(axis=1) / np.sqrt(accuracy_conds.shape[1])
    
    fix,ax = plt.subplots()
    ax.plot(range(len(phi_conds)),accuracy_conds,color=mcolors.CSS4_COLORS['silver'])
    ax.errorbar(range(len(phi_conds)),accuracy_conds.mean(axis=1),yerr = sem,color='black',linewidth=2)
    plt.xticks(range(len(phi_conds)),labels = phi_conds)
    plt.ylabel('Accuracy')
    plt.xlabel('Phase Difference')
    plt.title(str(AM[j]))
    
    AM_med[:,j] = np.median(accuracy_conds,axis=1)
    AM_mad_sem[:,j] = spst.median_absolute_deviation(accuracy_conds,axis=1,scale=1.4826) / np.sqrt(accuracy_conds.shape[1])
    AM_avgs[:,j] = accuracy_conds.mean(axis=1)
    AM_sems[:,j] = sem
    AM_rav[:,j] = accuracy_conds[:,0]
    
    
cmap = plt.get_cmap('hot')
cmap_colors = cmap(np.linspace(0,0.8,len(AM)))   
fig, ax = plt.subplots()
#ax.plot(range(len(phi_conds)),AM_avgs)
for n in range(0,len(AM)):
    ax.errorbar(range(len(phi_conds)),AM_avgs[:,n],yerr = AM_sems[:,n],color=cmap_colors[n,:],linewidth=2)
plt.xticks(range(len(phi_conds)),labels = phi_conds)
plt.ylim((0.2,1))
plt.ylabel('Accuracy')
plt.xlabel('Phase Difference')
plt.title('Dichotic')
plt.legend(AM)

t_diff = np.zeros(AM_avgs.shape)
for m in range (len(AM)):
    t_diff[:,m] = (1/AM[m]) * phi_conds.T/360


fig, ax = plt.subplots()
ax.plot(t_diff,AM_avgs,marker='x',linewidth=0)
plt.ylabel('Accuracy')
plt.xlabel('Time Difference')
plt.title('Dichotic')
plt.ylim((0.2,1))
plt.legend(AM)

#plot Rav
fig,ax = plt.subplots()
for n in range(0,len(AM)):
    ax.plot(range(len(phi_conds)),AM_rav[:,n],color=cmap_colors[n,:])
plt.xticks(range(len(phi_conds)),labels = phi_conds)
plt.ylabel('Accuracy')
plt.xlabel('Phase Difference')
plt.title('Rav Dichotic')
plt.legend(AM)

fontsize = 25
conf_95 = spst.binom(n=20,p=1/3).interval(.95)[1]/20  #20 trials per condition
fig, ax =plt.subplots(figsize=(12,9))
#ax.plot(range(len(AM)), AM_avgs.T,linewidth=2)
for ph in range(len(phi_conds)):
    ax.errorbar(range(len(AM)), AM_med[ph,:],AM_mad_sem[ph,:],linewidth=2,label=str(int(phi_conds[ph]))+ u'\xb0')
#ax.fill_between(range(len(AM)), np.repeat(conf_95,len(AM)),0,color='grey',alpha=0.5)
plt.legend(fontsize=fontsize)
plt.title('Incoh Dichotic')
plt.xticks(range(len(AM)),labels=AM,fontsize=fontsize)
plt.ylabel('Accuracy',fontsize=fontsize)
plt.xlabel('Modulation Freq',fontsize=fontsize)
#plt.title('Phase: ' + str(phi_conds[ph]),fontsize=fontsize)
plt.ylim([0.2,1])
plt.yticks(ticks=[0.2, 0.5, 0.8, 1.0],fontsize=fontsize)



with open('AMphi_dichotic.pickle','wb') as f:
    pickle.dump([AM_avgs,AM_sems, AM_med, AM_mad_sem],f)



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

StimData = '../../../Stimuli/TemporalCoding/OnlineExp_PhiAM/StimData_90.mat'

stim_info = loadmat(StimData)
correct = stim_info['correct'].squeeze()
stim_AMs = stim_info['fms'].squeeze()
correct = correct -1

Results_fname = 'PhiAM_phi90_Rav_results.json'
with open(Results_fname) as f:
    results = json.load(f)
    
    
subjects = []
trialnum = np.zeros((len(correct),len(results)))
AM = np.zeros((len(correct),len(results)))
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
        AM[cur_t,k] = int(annot[annot.find(':')+1:annot.find('}')])
        resps[cur_t,k] = int(trial['button_pressed'])
        rt[cur_t,k] = trial['rt']
        cur_t +=1 
        


resps = resps+1

AM_conds = np.unique(AM)
AM = AM[:,0]
accuracy_conds = np.zeros((len(AM_conds), len(subjects)))
for ll in range(len(AM_conds)):
    mask = AM==AM_conds[ll]
    for m in range(0,len(subjects)):
        accuracy = np.sum(resps[mask,m] == correct[mask]) / len(correct[mask])
        accuracy_conds[ll,m] = accuracy
        
    
fig, ax = plt.subplots()
ax.plot(range(len(AM_conds)),accuracy_conds)
plt.xticks(range(len(AM_conds)),labels = AM_conds)
plt.ylabel('Accuracy')
plt.xlabel('Phase Difference')
#plt.legend(subjects)



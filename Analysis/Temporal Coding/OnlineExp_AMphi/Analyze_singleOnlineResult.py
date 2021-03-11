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

StimData = '../../../Stimuli/TemporalCoding/OnlineExp_AMphi/DEMO_StimData_128.mat'

stim_info = loadmat(StimData)
correct = stim_info['correct'].squeeze()
stim_phis = stim_info['phis'].squeeze()

Results_fname = 'Demo_AMphi_AM128_Rav_results.json'
with open(Results_fname) as f:
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
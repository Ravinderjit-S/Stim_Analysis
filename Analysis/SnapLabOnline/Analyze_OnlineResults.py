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


stim_info = loadmat('StimData.mat')
correct = stim_info['correct'].squeeze()
stim_phis = stim_info['phis'].squeeze()

Results_fname = 'Test_Rav_results.json'
with open(Results_fname) as f:
    results = json.load(f)
    
    
subj = results[8]
trialnum = np.array([])
phi = np.array([])
resps = np.array([])
rt = np.array([])
for trial in subj:
    if not 'annot' in trial:
        continue
    tiralnum = np.append(trialnum,trial['trialnum'])
    annot = trial['annot']
    phi = np.append(phi,int(annot[annot.find(':')+1:annot.find('}')]))
    resps = np.append(resps,int(trial['button_pressed']))
    rt = np.append(rt,trial['rt'])

resps = resps+1

phi_conds = np.unique(phi)
accuracy_conds = np.array([])
for cond in phi_conds:
    mask = phi==cond
    accuracy = np.sum(resps[mask]==correct[mask]) / np.sum(mask)
    accuracy_conds = np.append(accuracy_conds, accuracy)
    
fig, ax = plt.subplots()
ax.plot(range(len(phi_conds)),accuracy_conds)
plt.xticks(range(len(phi_conds)),labels = phi_conds)
plt.ylabel('Accuracy')
plt.xlabel('Phase Difference')



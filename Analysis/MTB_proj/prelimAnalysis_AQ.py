#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 16:38:12 2021

@author: ravinderjit
"""

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

data_loc_binding = '/media/ravinderjit/Data_Drive/Data/MTB_Behavior/MTB_beh/'
data_loc_cmr = '/media/ravinderjit/Data_Drive/Data/MTB_Behavior/CMR_beh/'
data_loc_AQ = '/media/ravinderjit/Data_Drive/Data/AQ/'

beh_bind = sio.loadmat(data_loc_binding + 'BindingBeh.mat',squeeze_me=True)
beh_cmr = sio.loadmat(data_loc_cmr + 'CMRclickMod.mat',squeeze_me=True)
AQ = sio.loadmat(data_loc_AQ + 'AQscores.mat',squeeze_me=True)


#%% Get Subjects in right order for Data
#AQ, CMR and Binding don't necessarily have Subjects in same order or have the same
#Subjects, so need to account for that. Only analyze Subjects who have done 
#all tasks.


Subj_bind = list(beh_bind['Subjects'])
Subj_cmr = list(beh_cmr['Subjects'])
Subj_AQ = list(AQ['Subjects'])


Subjects = list(set(Subj_AQ) & set(Subj_bind) & set(Subj_cmr)) #Subjects that have done CMR and Binding
Subjects.sort()

bind_sort = np.zeros(len(Subjects),dtype=int)
aq_sort = np.zeros(len(Subjects),dtype=int)
cmr_sort = np.zeros(len(Subjects),dtype=int)

for s in range(len(Subjects)):
    bind_sort[s] = Subj_bind.index(Subjects[s])
    aq_sort[s] = Subj_AQ.index(Subjects[s])
    cmr_sort[s] = Subj_cmr.index(Subjects[s])

acc_bind = beh_bind['acc']
aq_scores = AQ['Scores']
cmr = beh_cmr['CMR']

acc_bind = acc_bind[:,bind_sort]
aq_scores = aq_scores[:,aq_sort]
cmr = cmr[cmr_sort]


#%% 
aq_full = np.sum(aq_scores,axis=0)
consec_coh = acc_bind[1:2,:].mean(axis=0)
spaced_coh = acc_bind[5:6,:].mean(axis=0)

plt.figure()
plt.scatter(aq_full,cmr)
plt.ylabel('CMR (dB)')
plt.xlabel('AQ score')

fig,ax = plt.subplots(5,1)
for a in range(5):
    ax[a].scatter(aq_scores[a,:],cmr)


fig, ax = plt.subplots(2,1)
ax[0].scatter(aq_full,consec_coh)
ax[1].scatter(aq_full,spaced_coh)


fig, ax = plt.subplots(2,1)
ax[0].scatter(cmr,consec_coh)
ax[1].scatter(cmr,spaced_coh)






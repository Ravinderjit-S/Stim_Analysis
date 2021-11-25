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

beh_bind = sio.loadmat(data_loc_binding + 'BindingBeh.mat',squeeze_me=True)
beh_cmr = sio.loadmat(data_loc_cmr + 'CMRclickMod.mat',squeeze_me=True)


#%% Get Subjects in right order for Data
#CMR and Binding don't necessarily have Subjects in same order or have the same
#Subjects, so need to account for that. Only analyze Subjects who have done 
#both tasks

Subj_CMR = list(beh_cmr['Subjects'])
Subj_bind = list(beh_bind['Subjects'])

Subjects = list(set(Subj_CMR) & set(Subj_bind)) #Subjects that have done CMR and Binding
Subjects.sort()

cmr_sort = np.zeros(len(Subjects),dtype=int)
bind_sort = np.zeros(len(Subjects),dtype=int)

for s in range(len(Subjects)):
    cmr_sort[s] = Subj_CMR.index(Subjects[s])
    bind_sort[s] = Subj_bind.index(Subjects[s])

acc_bind = beh_bind['acc']
cmr = beh_cmr['CMR']

cmr = cmr[cmr_sort]
acc_bind = acc_bind[:,bind_sort]

#%% Plot simple relationships

# For binding:
# 0 = 19:20
# 1 = 17-20
# 2 = 15-20
# 3 = 13:20
# 4 = 1, 8, 14, 20
# 5 = 1, 4, 8, 12, 16, 20
# 6 = 1, 4, 6, 9, 12, 15, 17, 20

beh_conds = ['2','4','6','8', '4 spaced','6 spaced','8 spaced']

fig,ax = plt.subplots(4,2,sharey=True)
ax = np.reshape(ax,8)

bad_subjects = acc_bind[1,:] < 0.9

for cnd in range(len(beh_conds)):
    ax[cnd].scatter(cmr[~bad_subjects],acc_bind[cnd,~bad_subjects],color='tab:blue')
    ax[cnd].scatter(cmr[bad_subjects],acc_bind[cnd,bad_subjects],color='tab:orange')
    ax[cnd].set_title(beh_conds[cnd])
    
 


consec_coh = acc_bind[1:2,:].mean(axis=0)

plt.figure()
plt.scatter(cmr,consec_coh)
plt.ylabel('Consecutive Coherence Detection')
plt.xlabel('CMR')
plt.ylim([0,1.1])

spaced_coh = acc_bind[5:6,:].mean(axis=0)

plt.figure()
plt.scatter(cmr,spaced_coh)
plt.ylabel('Spaced Coherence Detection')
plt.xlabel('CMR')
plt.ylim([0,1.1])

plt.figure()
plt.scatter(consec_coh,spaced_coh)
plt.xlabel('Consecutive Coherence Detection')
plt.ylabel('Spaced Coherence Detection')
plt.ylim([0,1.1])

MTB_behloc = '/media/ravinderjit/Data_Drive/Data/MTB_Behavior/'
sio.savemat(MTB_behloc + 'CMR_Bind.mat',{'CMR':cmr, 'spacedCoh': spaced_coh,
                                         'consec_coh': consec_coh,'Subjects':Subjects})




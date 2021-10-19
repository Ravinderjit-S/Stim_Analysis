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

plt.figure()
plt.scatter(cmr,acc_bind[1])
plt.xlabel('CMR (dB)')
plt.ylabel('Accurary')
plt.ylim([0,1.2])
plt.title('Binding 4 cont')

plt.figure()
plt.scatter(cmr,acc_bind[2])
plt.xlabel('CMR (dB)')
plt.ylabel('Accurary')
plt.ylim([0,1.2])
plt.title('Binding 6 cont')


plt.figure()
plt.scatter(cmr,acc_bind[5])
plt.xlabel('CMR (dB)')
plt.ylabel('Accurary')
plt.ylim([0,1.2])
plt.title('Binding 6 spaced')

plt.figure()
plt.scatter(cmr,acc_bind[6])
plt.xlabel('CMR (dB)')
plt.ylabel('Accurary')
plt.ylim([0,1.2])
plt.title('Binding 8 spaced')








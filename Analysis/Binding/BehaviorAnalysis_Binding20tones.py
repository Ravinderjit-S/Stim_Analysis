#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 28 13:13:36 2021

@author: ravinderjit
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import os
from matplotlib import rcParams

data_loc = '/media/ravinderjit/Data_Drive/Data/MTB_Behavior/MTB_beh/'
fig_loc = '/media/ravinderjit/Data_Drive/Data/Figures/MTB/'

Subjects = ['S211', 'S246', 'S259', 'S268', 'S269', 'S270', 'S271', 'S272',
 'S273', 'S274', 'S277', 'S279', 'S280', 'S281', 'S282', 'S284', 'S285', 'S288','S290',
 'S207', 'S303', 'S305', 'S078']

acc = np.zeros((7,len(Subjects)))
for sub in range(len(Subjects)):
    subject = Subjects[sub]
    dat_subpath = os.path.join(data_loc,subject + '_BindingBehavior20tones.mat')
    data = sio.loadmat(dat_subpath)
    
    CorrSet = data['CorrSet'][0]
    Corr_inds = data['Corr_inds'][0]
    correctList = data['correctList'][0]
    respList = data['respList'][0]
    
    ntrials = np.sum(CorrSet==1) #20
    

    for cond in np.unique(CorrSet):
        mask = CorrSet==cond 
        num_right = np.sum(correctList[mask] == respList[mask])
        acc[cond-1,sub] = num_right/ntrials
        
        
  
# 0 = 19:20
# 1 = 17-20
# 2 = 15-20
# 3 = 13:20
# 4 = 1, 8, 14, 20
# 5 = 1, 4, 8, 12, 16, 20
# 6 = 1, 4, 6, 9, 12, 15, 17, 20
        
rcParams.update({'font.size': 15})

mean_acc = acc.mean(axis=1)    
se_acc = acc.std(axis=1) / np.sqrt(acc.shape[1])    
        
plt.figure()
plt.bar([0,0.7],mean_acc[1:3],  width = 0.3, yerr= se_acc[1:3],color='k')
plt.bar([0.3,1],mean_acc[4:6],  width = 0.3, yerr = se_acc[4:6],color='grey')
plt.ylim([0,1])
plt.xticks([0,0.3,0.7,1],labels=['1.5 ERBs', '6.3 ERBs', '1.5 ERBs','5.7 ERBs'])
plt.ylabel('Accuracy')
plt.savefig(fig_loc + 'BindingAccuracy.svg',format='svg')

plt.figure()
plt.plot([1,2,3,4],acc[:4,:])

plt.figure()
plt.plot([1,2,3],acc[4:,:])

plt.figure()
plt.plot(np.arange(7)+1,acc)


sio.savemat(data_loc + 'BindingBeh.mat',{'acc':acc, 'Subjects':Subjects})


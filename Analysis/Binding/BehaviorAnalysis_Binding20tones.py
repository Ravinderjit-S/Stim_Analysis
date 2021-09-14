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


data_loc = '/media/ravinderjit/Data_Drive/Data/BehaviorData/MTB/BindingBeh20tones/'

Subjects = ['S211','S246','SVM']

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
        
        
  

# 1 = 17-20
# 2 = 15-20
# 4 = 1, 8, 14, 20
# 5 = 1, 4, 8, 12, 16, 20

mean_acc = acc.mean(axis=1)    
se_acc = acc.std(axis=1) / np.sqrt(acc.shape[1])    
        
plt.figure()
plt.bar([0,0.7],mean_acc[1:3],  width = 0.3, yerr= se_acc[1:3])
plt.bar([0.3,1],mean_acc[4:6],  width = 0.3, yerr = se_acc[4:6])
plt.ylim([0,1])
plt.xticks([0,0.3,0.7,1],labels=['1.5 ERBs', '6.3 ERBs', '1.5 ERBs','3.8 ERBs'])
plt.ylabel('Accuracy')



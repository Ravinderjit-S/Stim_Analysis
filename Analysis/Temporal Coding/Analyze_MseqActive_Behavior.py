#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 09:29:33 2021

@author: ravinderjit
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

data_loc = '/media/ravinderjit/Data_Drive/Data/BehaviorData/Mseq_ActiveBehavior/'

Subjects = ['S207', 'S228','S236','S238','S239','S246','S247','S250']
Subjects_sd = ['S207', 'S228', 'S236', 'S238', 'S239', 'S250'] #Leaving out S211 for now

acc_count = np.zeros(len(Subjects))
acc_sd = np.zeros(len(Subjects_sd))

for sub in range(len(Subjects)):
    subject = Subjects[sub]
    data = sio.loadmat(data_loc + subject + '_AMmseqActive.mat')
    acc_count[sub] = np.sum(data['correctList'] == data['respList'])
    
for sub in range(len(Subjects_sd)):
    subject = Subjects_sd[sub]
    data = sio.loadmat(data_loc + subject + '_AMmseq_shiftDetect.mat')
    acc_sd[sub] = np.sum(data['correctList'] == data['respList'])
    





#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 14:56:46 2023

@author: ravinderjit
"""


import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import mne
from anlffr.preproc import find_blinks
from EEGpp import EEGconcatenateFolder
from mne.preprocessing.ssp import compute_proj_epochs
import os
import pickle
import sys
sys.path.append(os.path.abspath('../mseqAnalysis/'))
from mseqHelper import mseqXcorr
from mseqHelper import mseqXcorrEpochs_fft
from sklearn.decomposition import PCA


data_loc = '/media/ravinderjit/Data_Drive/Data/EEGdata/MTB/mTRF/'
pickle_loc = data_loc + 'Pickles/'

Subjects = ['S069', 'S072', 'S078', 'S088', 'S104', 'S105', 'S259', 'S260', 'S268', 'S269',
            'S270', 'S271', 'S273', 'S274', 'S277', 'S279', 'S280', 'S281', 'S282', 'S284', 
            'S285', 'S288', 'S290', 'S291', 'S303', 'S305', 'S308', 'S310', 'S312', 'S337', 
            'S339', 'S340', 'S341', 'S342', 'S344', 'S345', 'S347', 'S352', 'S355', 'S358']





age = np.array([49, 55, 47, 52, 51, 61, 20, 33, 19, 19, 
       21, 21, 18, 19, 20, 20, 20, 21, 19, 26,
       19, 30, 21, 66, 28, 27, 59, 70, 37, 66,
       71, 39, 35, 54, 60, 61, 38, 35, 49, 56 ])


Acz_evk = np.empty([8192,len(Subjects),])
for s in range(len(Subjects)):
    subject = Subjects[s]
    with open(os.path.join(pickle_loc,subject +'_AMmseq10bits_epochs.pickle'),'rb') as file:
        if s ==25:
            [Ht_epochs,t_epochs, refchans, ch_picks] = pickle.load(file)
        else:
            [Ht_epochs,t_epochs, refchans] = pickle.load(file)
        Cz_evkd = Ht_epochs[-1,:,:].mean(axis=0)
    t_0 = np.where(t_epochs>=0)[0][0]
    t_15 = np.where(t_epochs>=0.015)[0][0]
    Cz_evkd = Cz_evkd - Cz_evkd[t_0]
    Cz_evkd = Cz_evkd / np.max(Cz_evkd[t_0:t_15])
    Acz_evk[:,s] = Cz_evkd
    
        
        
        
plt.figure()
plt.plot(t_epochs,Acz_evk)
plt.plot(t_epochs,Acz_evk.mean(axis=1),color='k',linewidth=2)
plt.xlim([-0.05,0.5])

t_cuts = [.016, .033, .066, .123, .268 ]
pca = PCA(n_components=2)


age_1 = age <25 
age_2 = np.logical_and(age >=25, age < 36)
age_3 = np.logical_and(age >35, age <56)
age_4 = age >=56

mean_1 = Acz_evk[:,age_1].mean(axis=1)
mean_2 = Acz_evk[:,age_2].mean(axis=1)
mean_3 = Acz_evk[:,age_3].mean(axis=1)
mean_4 = Acz_evk[:,age_4].mean(axis=1)

std_err1 = Acz_evk[:,age_1].std(axis=1) / np.sqrt(np.sum(age_1))
std_err2 = Acz_evk[:,age_2].std(axis=1) / np.sqrt(np.sum(age_2))
std_err3 = Acz_evk[:,age_3].std(axis=1) / np.sqrt(np.sum(age_3))
std_err4 = Acz_evk[:,age_4].std(axis=1) / np.sqrt(np.sum(age_4))

plt.figure()
plt.plot(t_epochs, mean_1,label='< 25', color='b')
plt.fill_between(t_epochs, mean_1 - std_err1, mean_1 + std_err1, alpha=0.5, color='b')

plt.plot(t_epochs, mean_2,label='25-35', color='g')
plt.fill_between(t_epochs, mean_2 - std_err2, mean_2 + std_err2, alpha=0.5, color='g')

plt.plot(t_epochs, mean_3,label='35-55', color='orange')
plt.fill_between(t_epochs, mean_3 - std_err3, mean_3 + std_err3, alpha=0.5, color='orange')

plt.plot(t_epochs, mean_4,label='>56', color='r')
plt.fill_between(t_epochs, mean_4 - std_err4, mean_4 + std_err4, alpha=0.5, color='r')

plt.xlim([-.05,0.5])
plt.legend()


age_yng = np.logical_or(age_1, age_2)







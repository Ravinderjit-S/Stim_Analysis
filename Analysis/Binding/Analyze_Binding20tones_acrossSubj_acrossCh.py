#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 16:26:41 2024

@author: ravinderjit
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
import os
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy import signal
import mne
from scipy import stats

import statsmodels.api as sm
import statsmodels.formula.api as smf

#%% Functions

def sortSubs(SubjectsToSort, Subjects):
    #retun indexes of SubjectsToSort so that in same order as Subjects. Assumes all subjects are in SubjectsToSort
    
    indexes = np.zeros(len(Subjects),dtype=int)
    for s in range(len(Subjects)):
        indexes[s] = SubjectsToSort.index(Subjects[s])

    return indexes

def sortSubs_exception(SubjectsToSort, Subjects):
    #retun indexes of SubjectsToSort so that in same order as Subjects. Assumes all subjects are in SubjectsToSort
    #added exception for instance where subject is missing
    
    #only using for MRT currently
    
    indexes = np.zeros(len(Subjects),dtype=int)
    for s in range(len(Subjects)):
        if Subjects[s] in SubjectsToSort:
            indexes[s] = SubjectsToSort.index(Subjects[s])
        else:
            indexes[s] = -10
        
    return indexes


#%%Data Params


data_loc = '/media/ravinderjit/Data_Drive/Data/EEGdata/MTB/Binding'
pickle_loc = data_loc + '/Pickles/'

fig_loc =  '/media/ravinderjit/Data_Drive/Data/Figures/MTBproj/Binding/'

Subjects = [ 'S069', 'S072','S078','S088', 'S104', 'S105', 'S207','S211',
            'S259', 'S260', 'S268', 'S269', 'S270','S271', 'S272', 'S273',
            'S274', 'S277','S279', 'S280', 'S282', 'S284', 'S285', 'S288',
            'S290' ,'S281','S291', 'S303', 'S305', 'S308', 'S309', 'S310',
            'S312', 'S339', 'S340', 'S341', 'S344', 'S345', 'S337', 'S352',
            'S355', 'S358']


age = [49, 55, 47, 52, 51, 61, 25, 28, 20, 33, 19, 19, 21, 21, 20, 18,
       19, 20, 20, 20, 19, 26, 19, 30, 21, 21, 66, 28, 27, 59, 33, 70,
       37, 71, 39, 35, 60, 61, 66, 35, 49, 56]


A_epochs = []
A_evkd = []

ind = Subjects.index('S281')
age = age[ind]
Subjects = [Subjects[ind]]

#%% Load 

for subject in Subjects:
    print('Loading ' + subject)
    with open(os.path.join(pickle_loc,subject+'_Binding.pickle'),'rb') as file:
       [t, t_full, conds_save, epochs_save,evkd_save] = pickle.load(file)
        
    
    A_epochs.append(epochs_save)
    A_evkd.append(evkd_save)
    
    
#%% 32 Channel responses

# evkd_12 = []
# evkd_20 = []
# for sub in range(len(Subjects)):
#     evkd_12.append(A_evkd[sub][6].data)
#     evkd_20.append(A_evkd[sub][7].data)
    

# ga_12 = np.zeros((32,t_full.size))
# ga_20 = np.zeros((32,t_full.size))

# ch_div = np.zeros(32)

# for sub in range(len(Subjects)):
#     chs_keep = np.arange(32)
#     if Subjects[sub] == 'S273':
#         chs_keep = np.delete(chs_keep,[0,23,29])
#     elif Subjects[sub] == 'S271':
#         chs_keep = np.delete(chs_keep,[2,5,15,0,29])
#     elif Subjects[sub] == 'S284':
#         chs_keep = np.delete(chs_keep,[5,23,24,27,2])
        
#     for ch in range(32):
#         if np.any(chs_keep == ch):
#             ch_div[ch] += 1 
    
#     ga_12[chs_keep,:] += evkd_12[sub][chs_keep]
#     ga_20[chs_keep,:] += evkd_20[sub][chs_keep] 
    
# ga_12 /= ch_div[:,np.newaxis]
# ga_20 /= ch_div[:,np.newaxis]
        
# t_1 = np.where(t_full >=1.5)[0][0]
# t_2 = np.where(t_full >=2.0)[0][0]

# pca = PCA(n_components=2)
# pca.fit_transform(ga_12[:,t_1:t_2].T)
# pca.explained_variance_ratio_    

# coeffs = pca.components_
    
# vmin = coeffs.mean() - 2*coeffs.std()
# vmax = coeffs.mean() + 2*coeffs.std()

# plt.figure()
# mne.viz.plot_topomap(coeffs[0,:], mne.pick_info(A_evkd[0][7].info,np.arange(32)),vmin=vmin,vmax=vmax)


#%% Look for source with PCA

    
  
#%% Get evoked responses
    
A_evkd_cz = np.zeros((len(t),len(Subjects),len(conds_save)-2))

for sub in range(len(Subjects)):
    for cond in range(len(conds_save)-2):
        A_evkd_cz[:,sub,cond] = A_epochs[sub][cond].mean(axis=0)
        
        

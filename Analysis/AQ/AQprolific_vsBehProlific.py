#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 12 00:57:25 2022

@author: ravinderjit
"""


import pandas as pd
import numpy as np
import os
import scipy.io as sio
import matplotlib.pyplot as plt

data_loc = '/home/ravinderjit/Documents/Data/AQ_prolific/'
cmr_file = 'cmr_hari_results.csv'
fodl_file = 'F0DLs_results.csv'
Gap4k_file = 'Gap4KHz_results.csv'

#%% Load Data
data_cmr = pd.read_csv(os.path.join(data_loc,cmr_file))
data_fodl =  pd.read_csv(os.path.join(data_loc,fodl_file))
data_gap4k = pd.read_csv(os.path.join(data_loc,Gap4k_file))

#%% Get CMR data

Subjects_cmr = data_cmr['subj'].to_numpy()
age_cmr = data_cmr['age'].to_numpy()

Subjects_cmr, ind = np.unique(Subjects_cmr,return_index=True)
age_cmr = age_cmr[ind]

acc_cmr = np.zeros([12,len(Subjects_cmr)])

for s in range(len(Subjects_cmr)):
    acc_cmr[:,s] = data_cmr['score'][data_cmr['subj'] == Subjects_cmr[s]][6:]

snrs_cmr = data_cmr['snr'][data_cmr['subj'] == Subjects_cmr[0]][6:]
conds_cmr = data_cmr['Condition'][data_cmr['subj'] == Subjects_cmr[0]][6:]

#%% Get F0dl data

Subjects_fodl = data_fodl['subj'].to_numpy()
age_fodl = data_fodl['age']

Subjects_fodl, ind = np.unique(Subjects_fodl,return_index=True)
age_fodl = age_cmr[fodl]








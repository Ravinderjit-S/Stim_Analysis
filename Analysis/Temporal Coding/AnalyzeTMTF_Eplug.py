#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 18:35:14 2021

@author: ravinderjit
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import pickle





data_loc = '/media/ravinderjit/Data_Drive/Data/EEGdata/TemporalCoding/TMTF/'
pickle_loc = data_loc + 'Pickles/'

Subjects = ['E001_1', 'E001_2','E002_Visit_1','E002_Visit_2','E002_Visit_3',
            'E003', 'E004_Visit_1','E004_Visit_2','E004_Visit_3', 'E005_Visit_1',
            'E005_Visit_2', 'E006_Visit_1', 'E006_Visit_2', 'E006_Visit_3',
            'E007_Visit_1', 'E007_Visit_2', 'E007_Visit_3', 'E012_Visit_1', 'E012_Visit_2' ,
            'E012_Visit_3', 'E014', 'E016', 'E022_Visit_1', 'E022_Visit_2', 'E022_Visit_3',
            ]


A_Ht = []

for sub in range(len(Subjects)):
    subject = Subjects[sub]
    with open(os.path.join(pickle_loc,subject+'_TMTF_cz.pickle'),'rb') as file:
       [t, Ht, info_obj, ch_picks] = pickle.load(file)
       
   
    A_Ht.append(Ht)
    
#%% put together two responses. They were played with opposite polarity
# so can put together here 
for sub in range(len(A_Ht)):
    A_Ht[sub] = np.concatenate((A_Ht[sub][0], A_Ht[sub][1]),axis=0)
    
#%% Look at 3 visits
    
S_v1 = [0,2,5,6,9,11,14,17,20,21,22]
S_v2 = [1,3,np.nan,7,10,12,15,18,np.nan,np.nan,23]
S_v3 = [np.nan, 4, np.nan,8,np.nan,13,16,19,np.nan,np.nan,24]

Subjects_v1 = ['E001', 'E002', 'E003', 'E004', 'E005', 'E006', 'E007', 'E012', 
               'E014', 'E016', 'E022']

fig,ax = plt.subplots(2,6,sharex=True)
ax = np.reshape(ax,12)

t_0 = np.where(t>=0)[0][0]

V1_h = []
V2_h = []
V3_h = []



for s in range(len(S_v1)):
    Ht_s = A_Ht[S_v1[s]] 
    s_mean = Ht_s.mean(axis=0)
    s_mean = s_mean - s_mean[t_0]
    s_sem = Ht_s.std(axis=0) / np.sqrt(Ht_s.shape[0])
    ax[s].plot(t,s_mean,color='tab:blue')
    ax[s].fill_between(t,s_mean-s_sem,s_mean+s_sem,alpha=0.5)
    ax[s].set_title(Subjects_v1[s])
    V1_h.append(s_mean)
    
    
    if ~np.isnan(S_v2[s]):
        Ht_s2 = A_Ht[S_v2[s]]
        s2_mean = Ht_s2.mean(axis=0)
        s2_mean = s2_mean - s2_mean[t_0]
        s2_sem = Ht_s2.std(axis=0) / np.sqrt(Ht_s2.shape[0])
        ax[s].plot(t,s2_mean,color='tab:orange')
        ax[s].fill_between(t,s2_mean-s2_sem,s2_mean+s2_sem,alpha=0.5)
        V2_h.append(s2_mean)
        
        
    if ~np.isnan(S_v3[s]):
        Ht_s3 = A_Ht[S_v3[s]]
        s3_mean = Ht_s3.mean(axis=0)
        s3_mean = s3_mean - s3_mean[t_0]
        s3_sem = Ht_s3.std(axis=0) / np.sqrt(Ht_s3.shape[0])
        ax[s].plot(t,s3_mean,color='tab:green')
        ax[s].fill_between(t,s3_mean-s3_sem,s3_mean+s3_sem,alpha=0.5)
        V3_h.append(s3_mean)

ax[0].set_xlim(-0.050,0.1)
ax[0].set_xlabel('Time (sec)')

V1_h = np.array(V1_h)
V2_h = np.array(V2_h)
V3_h= np.array(V3_h)

V1_h = np.delete(V1_h,7,axis=0)

V1_avg = V1_h.mean(axis=0)
V2_avg = V2_h.mean(axis=0)
V3_avg = V3_h.mean(axis=0)

V1_sem = V1_h.std(axis=0) / np.sqrt(V1_h.shape[0])
V2_sem = V2_h.std(axis=0) / np.sqrt(V2_h.shape[0])
V3_sem = V3_h.std(axis=0) / np.sqrt(V3_h.shape[0])

plt.figure()
plt.plot(t,V1_avg,label='V1')
plt.fill_between(t,V1_avg-V1_sem,V1_avg+V1_sem,alpha=0.5)
plt.plot(t,V2_avg,label='V2')
plt.fill_between(t,V2_avg-V2_sem,V2_avg+V2_sem,alpha=0.5)
plt.plot(t,V3_avg,label='V3')
plt.fill_between(t,V3_avg-V3_sem,V3_avg+V3_sem,alpha=0.5)
plt.xlim([0,0.1])
plt.xlabel('Time')
plt.legend()


#%% Look at F domain








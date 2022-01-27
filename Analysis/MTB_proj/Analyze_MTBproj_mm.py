#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 23 19:38:11 2022

@author: ravinderjit
get data set for a mixed model

"""

import os
import numpy as np
import scipy.io as sio
from numpy import matlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler



save_loc = '/media/ravinderjit/Data_Drive/Data/MTB_proj/'

Subjects = [ 'S072', 'S078', 'S088', 'S207', 'S211','S246', 'S259', 'S260',
            'S268', 'S269', 'S270', 'S271', 'S272', 'S273', 'S274', 'S277',
            'S279', 'S280', 'S281', 'S282', 'S284', 'S285', 'S288', 'S290',
            'S291', 'S303', 'S305', 'S308', 'S309', 'S310'] 


age = [55, 47, 52, 25, 28, 26, np.nan, 33, 19, 19, 21, 21, 20, 18, 19, 20, 20, 
       20, 21, 19, 26, 19, 30, 21, 66, 28, 27, 59, 33, 70]

age_class = []
for a in age:
    if np.isnan(a):
        age_class.append(0) #i think its a young person
    
    elif (a <= 35):
        age_class.append(0)
    elif((a >35) & (a <=55)):
        age_class.append(1)
    else:
        age_class.append(2)
            
        
    

#%% CMR data

data_loc_cmr = '/media/ravinderjit/Data_Drive/Data/MTB_Behavior/CMR_beh/'

sid_mm = np.array([])
SNR_mm = np.array([])
coh_mm = np.array([])
acc_mm = np.array([])
age_mm = np.array([])

for sub in range(len(Subjects)):

    subject = Subjects[sub]
    data = sio.loadmat(data_loc_cmr + subject + '_CMRrandModClicky.mat',squeeze_me=True)
    ntrials = 20
    
    #SNRs_0 = np.concatenate((np.array([-10]),np.arange(-25,-55,-5)))
    #SNRs_1 = np.concatenate((np.array([-20]),np.arange(-37,-61,-3)))
    
    SNRs_0 = data['SNR_0']
    SNRs_1 = data['SNR_1']
    
    snrList_unique = np.concatenate((SNRs_0,SNRs_1))
    snrList = matlib.repmat(snrList_unique,1,20).squeeze()
    
    cohList = data['cohList']
    correctList = data['correctList']
    respList = data['respList']
    
    acc_0 = np.zeros([SNRs_0.size])
    acc_1 = np.zeros([SNRs_1.size])
    
    for ii in range(acc_0.size):
        mask = (cohList == 0) & (snrList == SNRs_0[ii])
        acc_0[ii] = np.sum(correctList[mask] == respList[mask]) / ntrials
        
    for ii in range(acc_1.size):
        mask = (cohList ==1) & (snrList == SNRs_1[ii])
        acc_1[ii] = np.sum(correctList[mask] == respList[mask]) / ntrials


    sid_mm = np.concatenate((sid_mm, np.repeat(sub,SNRs_0.size + SNRs_1.size) ))
    age_mm = np.concatenate((age_mm, np.repeat(age[sub],SNRs_0.size + SNRs_1.size) ))
    
    SNR_mm = np.concatenate((SNR_mm,SNRs_0,SNRs_1))
    acc_mm = np.concatenate((acc_mm,acc_0, acc_1))
    coh_mm = np.concatenate((coh_mm, np.zeros(SNRs_0.size), np.ones(SNRs_1.size)  ))
    


sio.savemat(save_loc + 'CMR_mm.mat',{'sid': sid_mm, 'age': age_mm, 
                                     'SNR': SNR_mm, 'acc': acc_mm,
                                     'coh': coh_mm})  
    
    
    
#%% Analyze Binding beh

data_loc_mtbBeh = '/media/ravinderjit/Data_Drive/Data/MTB_Behavior/MTB_beh/'

sid_mm = np.array([])
dist_mm = np.array([])
ncoh_mm = np.array([])
acc_mm = np.array([])
age_mm = np.array([])

acc = np.zeros((7,len(Subjects)))
dists = [1.5, 1.5, 1.5, 1.5,
         9.5, 5.7, 4.1]

n_coh = [2, 4, 6, 8, 
         4, 6, 8]

for sub in range(len(Subjects)):
    subject = Subjects[sub]
    dat_subpath = os.path.join(data_loc_mtbBeh,subject + '_BindingBehavior20tones.mat')
    data = sio.loadmat(dat_subpath)
    
    CorrSet = data['CorrSet'][0]
    Corr_inds = data['Corr_inds'][0]
    correctList = data['correctList'][0]
    respList = data['respList'][0]
    
    ntrials = np.sum(CorrSet==1) #20
    
    acc_sub = np.zeros(7)
    
    for cond in np.unique(CorrSet):
        mask = CorrSet==cond 
        num_right = np.sum(correctList[mask] == respList[mask])
        acc[cond-1,sub] = num_right/ntrials
        acc_sub[cond-1] = num_right/ntrials
    
    sid_mm = np.concatenate((sid_mm, np.repeat(sub,len(acc_sub))))
    age_mm = np.concatenate((age_mm, np.repeat(age[sub],len(acc_sub)) ))
    dist_mm = np.concatenate((dist_mm, dists))
    ncoh_mm = np.concatenate((ncoh_mm, n_coh))
    acc_mm = np.concatenate((acc_mm, acc_sub))
    
    
        
sio.savemat(save_loc + 'BindBeh_mm.mat',{'sid': sid_mm, 'age': age_mm, 
                                     'dist': dist_mm, 'acc': acc_mm,
                                     'ncoh': ncoh_mm})  
  
# 0 = 19:20
# 1 = 17-20
# 2 = 15-20
# 3 = 13:20
# 4 = 1, 8, 14, 20
# 5 = 1, 4, 8, 12, 16, 20
# 6 = 1, 4, 6, 9, 12, 15, 17, 20



#%% CMR vs Binding


data_loc_cmr = '/media/ravinderjit/Data_Drive/Data/MTB_Behavior/CMR_beh/CMRclickMod.mat'
data_loc_mtb_beh = '/media/ravinderjit/Data_Drive/Data/MTB_Behavior/MTB_beh/BindingBeh.mat'
data_loc_JANE = '/media/ravinderjit/Data_Drive/Data/MTB_Behavior/SIN_Info_JANE/SINinfo_Jane.mat'
data_loc_MRT = '/media/ravinderjit/Data_Drive/Stim_Analysis/Analysis/SnapLabOnline/MTB_MRT_online/MTB_MRT.mat'


Sub_memr = ['S072','S078','S088','S207','S259','S260','S270', 
    'S271','S273','S274','S277','S281','S282','S285','S291', 
    'S305', 'S308', 'S309', 'S310']

memr_thresh = [70,82,67,58,67,54,70,64,64,64,70,58,73,67,82,46,55,67,64]


sid_mm = np.array([])
sc_mm = np.array([])
acc_mm = np.array([])
cmr_mm = np.array([])
lapse_mm = np.array([]) #cmr lapse
age_mm = np.array([])
jane_mm = np.array([])
mrt_mm = np.array([])
mrt_lapse_mm = np.array([])
memr_mm = np.array([])

cmr = sio.loadmat(data_loc_cmr,squeeze_me = True)
mtb_beh = sio.loadmat(data_loc_mtb_beh,squeeze_me = True)
jane = sio.loadmat(data_loc_JANE,squeeze_me = True)
mrt = sio.loadmat(data_loc_MRT,squeeze_me = True)

mtb_subs = list(mtb_beh['Subjects'])
acc_bind = mtb_beh['acc']

cmr_subs = list(cmr['Subjects'])

mrt_subs = list(mrt['Subjects'])
jane_subs = list(jane['Subjects'])


for sub in range(len(Subjects)):
    
    sc_mm = np.concatenate((sc_mm, [0,1,2]))
    sid_mm = np.concatenate((sid_mm,np.repeat(sub,3)))
    age_mm = np.concatenate((age_mm,np.repeat(age_class[sub],3)))
    
    
    if Subjects[sub] in mtb_subs:
        sub_ind = mtb_subs.index(Subjects[sub])
        thresh_coh = acc_bind[0,sub_ind]
        consec_coh = acc_bind[1:4,sub_ind].mean(axis=0)
        spaced_coh = acc_bind[4:7,sub_ind].mean(axis=0)
        
        acc_mm = np.concatenate((acc_mm,np.array([thresh_coh,consec_coh,spaced_coh])))
        
    else:
        acc_mm = np.concatenate((acc_mm,np.repeat(np.nan,3)))
        
    
    if Subjects[sub] in cmr_subs:
        sub_ind = cmr_subs.index(Subjects[sub])
        lapse_sub = cmr['lapse'][sub_ind]
        cmr_s = cmr['CMR'][sub_ind]
        
        lapse_mm = np.concatenate((lapse_mm,np.repeat(lapse_sub,3)))
        cmr_mm = np.concatenate((cmr_mm,np.repeat(cmr_s,3)))
        
    else:
        cmr_mm = np.concatenate((cmr_mm,np.repeat(np.nan,3)))
            
        
    if Subjects[sub] in jane_subs:
        sub_ind = jane_subs.index(Subjects[sub])
        
        jane_s = jane['thresholds'][sub_ind]
        
        jane_mm = np.concatenate((jane_mm,np.repeat(jane_s,3)))
        
    else:
        jane_mm = np.concatenate((jane_mm,np.repeat(np.nan,3)))
        
        
    if Subjects[sub] in mrt_subs:
        sub_ind = mrt_subs.index(Subjects[sub])
        
        mrt_s = mrt['thresholds'][sub_ind]
        mrt_lapse = mrt['lapse'][sub_ind]
        
        mrt_mm = np.concatenate((mrt_mm,np.repeat(mrt_s,3)))
        mrt_lapse_mm = np.concatenate((mrt_lapse_mm,np.repeat(mrt_lapse,3)))
        
    else:
        mrt_mm = np.concatenate((mrt_mm,np.repeat(np.nan,3)))
        mrt_lapse_mm = np.concatenate((mrt_lapse_mm,np.repeat(np.nan,3)))
        
        
    if Subjects[sub] in Sub_memr:
        sub_ind = Sub_memr.index(Subjects[sub])
        
        memr_mm = np.concatenate((memr_mm,np.repeat(memr_thresh[sub_ind],3)))
        
    else:
        memr_mm = np.concatenate((memr_mm,np.repeat(np.nan,3)))
        
                            
    

sio.savemat(save_loc + 'Bind_CMR_mm.mat',{'sid': sid_mm, 'TCS': sc_mm,
                                          'acc': acc_mm, 'cmr': cmr_mm,
                                          'cmr lapse': lapse_mm, 'age': age_mm,
                                          'jane': jane_mm, 'mrt': mrt_mm,
                                          'mrt_lapse': mrt_lapse_mm,
                                          'memr': memr_mm })      
    

#%% Binding phys vs Beh LM

data_loc = '/media/ravinderjit/Data_Drive/Data/EEGdata/MTB/Model/Binding20TonesModelVars.mat' 
bind = sio.loadmat(data_loc,squeeze_me=True)

features = bind['features']



#%% Binding phys vs Beh mm

data_loc = '/media/ravinderjit/Data_Drive/Data/EEGdata/MTB/Model/Binding20TonesModelVars.mat' 

bind = sio.loadmat(data_loc,squeeze_me=True)

sid_mm = np.array([])
age_mm = np.array([])

type_mm = np.array([])
ccoh_mm = np.array([])

B12on_mm = np.array([])
B20on_mm = np.array([])
B12m_mm = np.array([])
B20m_mm = np.array([])

pca1_mm = np.array([])
pca2_mm = np.array([])

acc_mm = np.array([])

bind_subjects = bind['Subjects']

for sub in range(len(bind_subjects)):
    
    if bind_subjects[sub] not in Subjects:
        continue
    
    if (sub==5): #remove outlier
        continue
    
    if (sub==11): #remove outlier
        continue
    
    sub_ind = Subjects.index(bind_subjects[sub])
    
    sid_mm = np.concatenate((sid_mm,np.repeat(sub,2)))
    age_mm = np.concatenate((age_mm,np.repeat(age_class[sub_ind],2)))
    
    type_mm = np.concatenate((type_mm, [1,2]))
    
    consec_coh = 1 - bind['consecCoh'][sub]
    ccoh_mm = np.concatenate((ccoh_mm,np.repeat(consec_coh,2)))
    
    B12on, B20on, B12mn, B20mn = bind['features'][sub,:]
    
    B12on_mm = np.concatenate((B12on_mm,np.repeat(B12on,2)))
    B20on_mm = np.concatenate((B20on_mm,np.repeat(B20on,2)))
    B12m_mm = np.concatenate((B12m_mm,np.repeat(B12mn,2)))
    B20m_mm = np.concatenate((B20m_mm,np.repeat(B20mn,2)))
    
    pca1, pca2 = bind['pca_feats'][sub,:]
    
    pca1_mm = np.concatenate((pca1_mm,np.repeat(pca1,2)))
    pca2_mm = np.concatenate((pca2_mm,np.repeat(pca2,2)))

    thresh_coh = bind['threshCoh'][sub]
    spaced_coh = bind['spacedCoh'][sub]
    
    acc_mm = np.concatenate((acc_mm,[thresh_coh, spaced_coh]))
    
    
    
    
sio.savemat(save_loc + 'Bind_phys_mm.mat',{'sid': sid_mm, 'type': type_mm,
                                          'acc': acc_mm, 'lapse': ccoh_mm,
                                          'B12on': B12on_mm, 'B20on': B20on_mm,
                                          'B12m' : B12m_mm, 'B20m': B20m_mm,
                                          'pca1' : pca1_mm, 'pca2': pca2_mm,
                                          'age': age_mm })    

    
    
    
    
    




#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 13:21:20 2022

@author: ravinderjit
"""

import os
import numpy as np
import scipy.io as sio
from numpy import matlib



save_loc = '/media/ravinderjit/Data_Drive/Data/MTB_proj/'

Subjects = [ 'S072', 'S078', 'S088', 'S207', 'S211','S246', 'S259', 'S260',
            'S268', 'S269', 'S270', 'S271', 'S272', 'S273', 'S274', 'S277',
            'S279', 'S280', 'S281', 'S282', 'S284', 'S285', 'S288', 'S290',
            'S291', 'S303', 'S305', 'S308', 'S309', 'S310'] 


age = [55, 47, 52, 25, 28, 26, 20, 33, 19, 19, 21, 21, 20, 18, 19, 20, 20, 
       20, 21, 19, 26, 19, 30, 21, 66, 28, 27, 59, 33, 70]

age_class = []
for a in age:
    if np.isnan(a):
        age_class.append(0) #i think its a young person
    
    elif (a <= 35):
        age_class.append(0)
    elif((a >35)):# & (a <=55)):
        age_class.append(1)
    # else:
    #     age_class.append(2)
            
        
data_loc_mtb_phys = '/media/ravinderjit/Data_Drive/Data/EEGdata/MTB/Model/Binding20TonesModelVars.mat'    
data_loc_mtb_beh = '/media/ravinderjit/Data_Drive/Data/MTB_Behavior/MTB_beh/BindingBeh.mat'        

#%% Make Binding and Binding phys Consec mm vars

mtb_beh = sio.loadmat(data_loc_mtb_beh,squeeze_me = True)
mtb_phys = sio.loadmat(data_loc_mtb_phys)

sid_mm = np.array([])
age_mm = np.array([])

acc_con = np.array([])
cond_con = np.array([])

B_mn1 = np.array([])
B_mn2 = np.array([])
B_on1 = np.array([])
B_on2 = np.array([])


beh_subjects = list(mtb_beh['Subjects'])
phys_subjects = list(mtb_phys['Subjects'])


for sub in range(len(Subjects)):
    
    sid_mm = np.concatenate((sid_mm,np.repeat(sub,4)))
    age_mm = np.concatenate((age_mm,np.repeat(age_class[sub],4)))
    
    if Subjects[sub] in beh_subjects:
        beh_ind = beh_subjects.index(Subjects[sub])
        acc_con = np.concatenate((acc_con, mtb_beh['acc'][:4,beh_ind]))
        cond_con = np.concatenate((cond_con,[2,4,6,8]))
    else:
        acc_con = np.concatenate((acc_con, np.repeat(np.nan,4)))
        cond_con = np.concatenate((cond_con, np.repeat(np.nan,4)))
        
    
    if Subjects[sub] in phys_subjects:
        phys_ind = phys_subjects.index(Subjects[sub])
        
        B_mn1 = np.concatenate((B_mn1, np.repeat( mtb_phys['feats_Bmn'][phys_ind,0] ,4)))
        B_mn2 = np.concatenate((B_mn2, np.repeat( mtb_phys['feats_Bmn'][phys_ind,1] ,4)))
        
        B_on1 = np.concatenate((B_on1, np.repeat( mtb_phys['feats_Bon'][phys_ind,0] ,4)))
        B_on2 = np.concatenate((B_on2, np.repeat( mtb_phys['feats_Bon'][phys_ind,1] ,4)))
        
    else:
        B_mn1 = np.concatenate((B_mn1, np.repeat( np.nan ,4)))
        B_mn2 = np.concatenate((B_mn2, np.repeat( np.nan ,4)))
        
        B_on1 = np.concatenate((B_on1, np.repeat( np.nan ,4)))
        B_on2 = np.concatenate((B_on2, np.repeat( np.nan ,4)))
            
    

    
sio.savemat(save_loc + 'BindConsec_phys_mm.mat',{'sid': sid_mm, 'age': age_mm,
                                                 'acc_con': acc_con, 'cond': cond_con,
                                                 'B_mn1':B_mn1, 'B_mn2':B_mn2,
                                                 'B_on1':B_on1, 'B_on2':B_on2,
                                             })    

    
    
#%% Make Binding and Binding phys Intervened mm vars
   
mtb_beh = sio.loadmat(data_loc_mtb_beh,squeeze_me = True)
mtb_phys = sio.loadmat(data_loc_mtb_phys)

sid_mm = np.array([])
age_mm = np.array([])

acc_inter = np.array([])
cond_inter = np.array([])

B_mn1 = np.array([])
B_mn2 = np.array([])
B_on1 = np.array([])
B_on2 = np.array([])


beh_subjects = list(mtb_beh['Subjects'])
phys_subjects = list(mtb_phys['Subjects'])


for sub in range(len(Subjects)):
    
    sid_mm = np.concatenate((sid_mm,np.repeat(sub,3)))
    age_mm = np.concatenate((age_mm,np.repeat(age_class[sub],3)))
    
    if Subjects[sub] in beh_subjects:
        beh_ind = beh_subjects.index(Subjects[sub])
        acc_inter = np.concatenate((acc_inter, mtb_beh['acc'][4:,beh_ind]))
        cond_inter = np.concatenate((cond_inter,[4,6,8]))
    else:
        acc_con = np.concatenate((acc_con, np.repeat(np.nan,3)))
        cond_con = np.concatenate((cond_con, np.repeat(np.nan,3)))
        
    
    if Subjects[sub] in phys_subjects:
        phys_ind = phys_subjects.index(Subjects[sub])
        
        B_mn1 = np.concatenate((B_mn1, np.repeat( mtb_phys['feats_Bmn'][phys_ind,0] ,3)))
        B_mn2 = np.concatenate((B_mn2, np.repeat( mtb_phys['feats_Bmn'][phys_ind,1] ,3)))
        
        B_on1 = np.concatenate((B_on1, np.repeat( mtb_phys['feats_Bon'][phys_ind,0] ,3)))
        B_on2 = np.concatenate((B_on2, np.repeat( mtb_phys['feats_Bon'][phys_ind,1] ,3)))
        
    else:
        B_mn1 = np.concatenate((B_mn1, np.repeat( np.nan ,3)))
        B_mn2 = np.concatenate((B_mn2, np.repeat( np.nan ,3)))
        
        B_on1 = np.concatenate((B_on1, np.repeat( np.nan ,3)))
        B_on2 = np.concatenate((B_on2, np.repeat( np.nan ,3)))    
    

    
sio.savemat(save_loc + 'BindInter_phys_mm.mat',{'sid': sid_mm, 'age': age_mm,
                                                 'acc_inter': acc_inter, 'cond': cond_inter,
                                                 'B_mn1':B_mn1, 'B_mn2':B_mn2,
                                                 'B_on1':B_on1, 'B_on2':B_on2,
                                             })    

    
    
    


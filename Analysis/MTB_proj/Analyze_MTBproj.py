#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 22 14:58:27 2022

@author: ravinderjit

Put together all pieces of processed data for the MTB project
"""


import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import pandas as pd

save_loc = '/media/ravinderjit/Data_Drive/Data/'

#%% Functions

def SortSubjects(Subjects,Subjects2):
    #Find indices of Subjects in Subjects 2
    index_sub = []
    del_inds = []
    for s in range(len(Subjects2)):
        if Subjects2[s] in Subjects:
            index_sub.append(Subjects.index(Subjects2[s]))
        else:
            del_inds.append(s)
    
    return index_sub, del_inds
    


#%% Load data


#data_loc_audiogram  .... get audiogram data

#Behavior
data_loc_cmr = '/media/ravinderjit/Data_Drive/Data/MTB_Behavior/CMR_beh/CMRclickMod.mat'
data_loc_mtb_beh = '/media/ravinderjit/Data_Drive/Data/MTB_Behavior/MTB_beh/BindingBeh.mat'
data_loc_JANE = '/media/ravinderjit/Data_Drive/Data/MTB_Behavior/SIN_Info_JANE/SINinfo_Jane.mat'
data_loc_MRT = '/media/ravinderjit/Data_Drive/Stim_Analysis/Analysis/SnapLabOnline/MTB_MRT_online/MTB_MRT.mat'

cmr = sio.loadmat(data_loc_cmr,squeeze_me = True)
mtb_beh = sio.loadmat(data_loc_mtb_beh,squeeze_me = True)
jane = sio.loadmat(data_loc_JANE,squeeze_me = True)
mrt = sio.loadmat(data_loc_MRT,squeeze_me = True)

#AQ and age
data_loc_aq = '/media/ravinderjit/Data_Drive/Data/AQ/AQscores.mat'

aq = sio.loadmat(data_loc_aq,squeeze_me=True)

Subjects_age = [ 'S072', 'S078', 'S088', 'S207', 'S211', 'S246', 'S259',
                'S260', 'S268', 'S269', 'S270', 'S271', 'S272', 'S273',
                'S274', 'S277', 'S279', 'S280', 'S281', 'S282', 'S284', 
                'S285', 'S288', 'S290', 'S291', 'S303', 'S305', 'S308',
                'S309', 'S310']
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


#Physiology
data_loc_mtb_phys = '/media/ravinderjit/Data_Drive/Data/EEGdata/MTB/Model/Binding20TonesModelVars.mat'
#data_loc_ACR = []
#data_loc_MEMR = []

mtb_phys = sio.loadmat(data_loc_mtb_phys)


#%% Put all data into a dataframe

#% Initialize data 
cmr_dat = np.empty(len(Subjects_age))
cmr_lapse = np.empty(len(Subjects_age))
thresh_coh = np.empty(len(Subjects_age))
consec_coh = np.empty(len(Subjects_age))
spaced_coh = np.empty(len(Subjects_age))
jane_dat = np.empty(len(Subjects_age))
mrt_dat = np.empty(len(Subjects_age))
aq_dat = np.empty(len(Subjects_age))
memr = np.empty(len(Subjects_age))

cmr_dat[:] = np.nan
cmr_lapse[:] = np.nan
thresh_coh[:] = np.nan
consec_coh[:] = np.nan
spaced_coh[:] = np.nan
jane_dat[:] = np.nan
mrt_dat[:] = np.nan
aq_dat[:] = np.nan
memr[:] = np.nan

data = pd.DataFrame(data={'Subjects': Subjects_age, 'age':age_class})

# Add CMR data
index_sort, del_inds = SortSubjects(Subjects_age, cmr['Subjects'])
cmr_ = cmr['CMR']
cmr_lap_ = cmr['lapse']
cmr_ = np.delete(cmr_,del_inds) # delete subjects in CMR not apart of MTB study
cmr_lap_ = np.delete(cmr_lap_,del_inds)

cmr_dat[index_sort] = cmr_ #sort subjects in case in different order in CMR saved data
cmr_lapse[index_sort] = cmr_lap_
data['CMR'] = cmr_dat
data['CMRlapse'] = cmr_lap_

# Add MTB beh data
acc_bind = mtb_beh['acc']
index_sort, del_inds = SortSubjects(Subjects_age, mtb_beh['Subjects'])
acc_bind = np.delete(acc_bind,del_inds,axis=1)

thresh_coh[index_sort] = acc_bind[0,:]
consec_coh[index_sort] = acc_bind[1:4,:].mean(axis=0)
spaced_coh[index_sort] = acc_bind[4:7,:].mean(axis=0)

data['thresh_coh'] = thresh_coh
data['consec_coh'] = consec_coh
data['spaced_coh'] = spaced_coh

# Add JANE
index_sort, del_inds = SortSubjects(Subjects_age, jane['Subjects'])
jane_dat[index_sort] = np.delete(jane['thresholds'],del_inds)
data['Jane'] = jane_dat

# Add MRT
index_sort, del_inds = SortSubjects(Subjects_age, mrt['Subjects'])
mrt_dat[index_sort] = np.delete(mrt['thresholds'], del_inds)
data['MRT'] = mrt_dat

# Add AQ
index_sort, del_inds = SortSubjects(Subjects_age, aq['Subjects'])
aq_scores = np.delete(aq['Scores'],del_inds,axis=1)

aq_dat[index_sort] = aq_scores.sum(axis=0)
data['aq'] = aq_dat

# Add MTB Phys
index_sort, del_inds = SortSubjects(Subjects_age,mtb_phys['Subjects'])

feat_labels = mtb_phys['feat_labels']
features = np.delete(mtb_phys['features'],del_inds,axis=1)
pca_feats = np.delete(mtb_phys['pca_feats'],del_inds,axis=0)

for ff in range(len(feat_labels)):
    feat = np.empty(len(Subjects_age))
    feat[:] = np.nan
    feat[index_sort] = features[ff,:]
    data[feat_labels[ff].strip()] = feat

pca_f1 = np.empty(len(Subjects_age))
pca_f2 = np.empty(len(Subjects_age))
pca_f1[:] = np.nan
pca_f2[:] = np.nan

pca_f1[index_sort] = pca_feats[:,0]
pca_f2[index_sort] = pca_feats[:,1]

data['PCA_f1'] = pca_f1
data['PCA_f2'] = pca_f2

#temp MEMR
Sub_memr = ['S072','S078','S088','S207','S259','S260','S270', 
    'S271','S273','S274','S277','S281','S282','S285','S291', 
    'S305', 'S308', 'S309', 'S310']

memr_thresh = [70,82,67,58,67,54,70,64,64,64,70,58,73,67,82,46,55,67,64]
index_sort, del_inds = SortSubjects(Subjects_age,Sub_memr)
memr[index_sort] = memr_thresh

data['memr'] = memr



#%% Plot Stuff

# Plot age stuff

fig,ax = plt.subplots(5,1,sharex=True)
ax[4].set_xlabel('AGE')

ax[0].scatter(data['age'], data['CMR'],color='tab:blue', label = 'CMR')
ax[0].set_ylabel('CMR')

ax[1].scatter(data['age'], data['consec_coh'], label= 'consec coh')
ax[1].scatter(data['age'], data['spaced_coh'], label='spaced coh')
ax[1].set_ylabel('Bind Acc')
ax[1].legend()

ax[2].scatter(data['age'], data['Jane'] )
ax[2].set_ylabel('Jane Thresh')

ax[3].scatter(data['age'], data['MRT'])
ax[3].set_ylabel('MRT Thresh')

ax[4].scatter(data['age'],data['memr'])
ax[4].set_ylabel('MEMR thresh')


# Plot CMR stuff

fig,ax = plt.subplots(3)
ax[2].set_xlabel('CMR')

ax[0].scatter(data['CMR'], (data['consec_coh'] + data['spaced_coh']) / 2)
#ax[0].scatter(data['CMR'],data['spaced_coh'])
ax[0].set_ylabel('Bind acc')

ax[1].scatter(data['CMR'], data['Jane'])
ax[1].set_ylabel('Jane Thresh')

ax[2].scatter(data['CMR'], data['MRT'])
ax[2].set_ylabel('MRT Thresh')

# Plot Binding stuff

fig,ax = plt.subplots(3)
ax[2].set_xlabel('bind acc')

bind_acc = (data['consec_coh'] + data['spaced_coh']) / 2

ax[0].scatter(bind_acc,data['Jane'])
ax[0].set_ylabel('Jane Thresh')

ax[1].scatter(bind_acc,data['MRT'])
ax[1].set_ylabel('MRT Thresh')

ax[2].scatter(bind_acc,data['consec_coh'])
ax[2].set_ylabel('consec_coh acc')

# Plot MEMR stuff

fig,ax = plt.subplots(3)
ax[2].set_xlabel('MEMR thresh')

ax[0].scatter(data['memr'],data['CMR'])
ax[0].set_ylabel('CMR')

ax[1].scatter(data['memr'],data['spaced_coh'])
ax[1].set_ylabel('Spaced acc')

ax[2].scatter(data['memr'],data['Jane'])
ax[2].set_ylabel('Jane')


# Plot MTB phys stuff

fig,ax = plt.subplots(4)
feat = data['PCA_f2']

ax[0].scatter(feat,data['spaced_coh'])
ax[0].set_ylabel('Spaced acc')

ax[1].scatter(feat,data['CMR'])
ax[1].set_ylabel('CMR')

ax[2].scatter(feat,data['Jane'])
ax[2].set_ylabel('Jane Thresh')

ax[3].scatter(feat,data['MRT'])
ax[3].set_ylabel('MRT thresh')


#%% Save as r data frame
import pyarrow.feather as feather

data['sub_ind'] = np.arange(30)
feather.write_feather(data, save_loc + 'MTB_dataframe')

#%% Convert to format for mixed effects analysis


#sio.savemat(save_loc + 'MTB_dataframe.mat',data)
















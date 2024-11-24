#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  7 11:10:36 2021

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

#%% Load data

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

    
  
#%% Get evoked responses
    
A_evkd_cz = np.zeros((len(t),len(Subjects),len(conds_save)-2))

for sub in range(len(Subjects)):
    for cond in range(len(conds_save)-2):
        A_evkd_cz[:,sub,cond] = A_epochs[sub][cond].mean(axis=0)
        
#%% Plot Average response across Subjects (unNormalized)

#Conds:
#   0 = 12 Onset
#   1 = 20 Onset
#   3 = 12AB
#   4 = 12BA
#   5 = 20AB
#   6 = 20BA         
        
cond_bind = ['12 Onset', '20 Onset', '12AB', '12BA', '20AB', '20BA', '12 all','20 all']
        
conds_comp = [[0,1], [2,4], [3,5]]
labels = ['Onset', 'Incoherent to Coherent', 'Coherent to Incoherent']
    
fig,ax = plt.subplots(3,1,sharex=True)
#fig.set_size_inches(20,10)
#plt.rcParams.update({'font.size': 26})



for jj in range(3):
    cnd1 = conds_comp[jj][0]
    cnd2 = conds_comp[jj][1]
    
    onset12_mean = (A_evkd_cz[:,:,cnd1]*1e6).mean(axis=1)
    onset12_sem = (A_evkd_cz[:,:,cnd1]*1e6).std(axis=1) / np.sqrt(A_evkd_cz.shape[1])
    
    ax[jj].plot(t,onset12_mean,label='12')  
    ax[jj].fill_between(t,onset12_mean - onset12_sem, onset12_mean + onset12_sem,alpha=0.5)
    
    onset20_mean = (A_evkd_cz[:,:,cnd2]*1e6).mean(axis=1)
    onset20_sem = (A_evkd_cz[:,:,cnd2]*1e6).std(axis=1) / np.sqrt(A_evkd_cz.shape[1])
    
    ax[jj].plot(t,onset20_mean,label='20')  
    ax[jj].fill_between(t,onset20_mean - onset20_sem, onset20_mean + onset20_sem,alpha=0.5)
    
    ax[jj].ticklabel_format(axis='y',style='sci',scilimits=(0,0))
    #ax[jj].set_title(labels[jj])
    ax[jj].tick_params(labelsize=12)


ax[0].legend(fontsize=12)
ax[2].set_xlabel('Time (s)',fontsize=14)
ax[2].set_ylabel('\u03bcV',fontsize=14)
ax[2].set_xlim([-0.050,1])
ax[2].set_xticks([0,0.2,0.4,0.6,0.8,1.0])
#ax[2].set_ylabel('$\mu$V')
#fig.suptitle('Average Across Participants')


plt.savefig(os.path.join(fig_loc,'All_12vs20_baselined.svg'),format='svg')

#%% Plot single subject like above

cond_bind = ['12 Onset', '20 Onset', '12AB', '12BA', '20AB', '20BA', '12 all','20 all']
        
conds_comp = [[0,1], [2,4], [3,5]]
labels = ['Onset', 'Incoherent to Coherent', 'Coherent to Incoherent']
    
fig,ax = plt.subplots(3,1,sharex=True)
#fig.set_size_inches(20,10)
#plt.rcParams.update({'font.size': 26})

Sub_to_plot = 'S273'
sub_ind_p = Subjects.index(Sub_to_plot)

for jj in range(3):
    cnd1 = conds_comp[jj][0]
    cnd2 = conds_comp[jj][1]
    
    onset12_mean = (A_evkd_cz[:,sub_ind_p,cnd1]*1e6)
    
    ax[jj].plot(t,onset12_mean,label='12')  
    ax[jj].fill_between(t,onset12_mean - onset12_sem, onset12_mean + onset12_sem,alpha=0.5)
    
    onset20_mean = (A_evkd_cz[:,:,cnd2]*1e6).mean(axis=1)
    onset20_sem = (A_evkd_cz[:,:,cnd2]*1e6).std(axis=1) / np.sqrt(A_evkd_cz.shape[1])
    
    ax[jj].plot(t,onset20_mean,label='20')  
    ax[jj].fill_between(t,onset20_mean - onset20_sem, onset20_mean + onset20_sem,alpha=0.5)
    
    ax[jj].ticklabel_format(axis='y',style='sci',scilimits=(0,0))
    #ax[jj].set_title(labels[jj])
    ax[jj].tick_params(labelsize=12)


ax[0].legend(fontsize=12)
ax[2].set_xlabel('Time (s)',fontsize=14)
ax[2].set_ylabel('\u03bcV',fontsize=14)
ax[2].set_xlim([-0.050,1])
ax[2].set_xticks([0,0.2,0.4,0.6,0.8,1.0])
#ax[2].set_ylabel('$\mu$V')
fig.suptitle(Sub_to_plot)

#%% Young Vs Old

cond_bind = ['12 Onset', '20 Onset', '12AB', '12BA', '20AB', '20BA', '12 all','20 all']
        
conds_comp = [[0,1], [2,4], [3,5]]
labels = ['Onset', 'Incoherent to Coherent', 'Coherent to Incoherent']
    
fig,ax = plt.subplots(3,1,sharex=True)

young = np.array(age) < 40
old = np.array(age) >= 40

conds_comp = [[1,1], [4,4], [3,3]]
labels = ['Onset', 'Incoherent to Coherent 20', 'Coherent to Incoherent 20']

for jj in range(3):
    cnd1 = conds_comp[jj][0]
    cnd2 = conds_comp[jj][1]
    
    onset12_mean = A_evkd_cz[:,young,cnd1].mean(axis=1)
    onset12_sem = A_evkd_cz[:,young,cnd1].std(axis=1) / np.sqrt(A_evkd_cz[:,young,cnd1].shape[1])
    
    ax[jj].plot(t,onset12_mean,label='Young',color='tab:green')  
    ax[jj].fill_between(t,onset12_mean - onset12_sem, onset12_mean + onset12_sem,alpha=0.5, color='tab:green')
    
    onset20_mean = A_evkd_cz[:,old,cnd2].mean(axis=1)
    onset20_sem = A_evkd_cz[:,old,cnd2].std(axis=1) / np.sqrt(A_evkd_cz[:,old,cnd2].shape[1])
    
    ax[jj].plot(t,onset20_mean,label='Old', color='tab:purple')  
    ax[jj].fill_between(t,onset20_mean - onset20_sem, onset20_mean + onset20_sem,alpha=0.5,color='tab:purple')
    
    ax[jj].ticklabel_format(axis='y',style='sci',scilimits=(0,0))
    ax[jj].set_title(labels[jj])


ax[0].legend()
ax[2].set_xlabel('Time')
#ax[2].set_ylabel('$\mu$V')
#fig.suptitle('Average Across Participants')

plt.savefig(os.path.join(fig_loc,'YoungvsOld_20.png'),format='png')


#%% Normalize Data to max of of abs of onset response

# for sub in range(len(Subjects)):
#      sub_norm = np.max(np.abs((A_epochs[sub][0].mean(axis=0) + A_epochs[sub][1].mean(axis=0)) / 2))
#      for cnd in range(len(cond_bind)):
#         A_epochs[sub][cnd] /= sub_norm

#%% Lets look at Full time

#t_f = np.arange(-0.3,5.5+1/4096,1/4096)
t_f = t_full

A_evkd_f = np.zeros((len(t_f),len(Subjects),2))

for sub in range(len(Subjects)):
    for cond in range(2):
        A_evkd_f[:,sub,cond] = A_epochs[sub][6+cond].mean(axis=0)

                 
sub_mean = A_evkd_f[:,:,0].mean(axis=1) * 1e6
sub_sem = (A_evkd_f[:,:,0]*1e6).std(axis=1) / np.sqrt(len(Subjects))

plt.figure()
plt.plot(t_f,sub_mean,label='12')
plt.fill_between(t_f, sub_mean-sub_sem,sub_mean+sub_sem,alpha=0.5)

sub_mean = A_evkd_f[:,:,1].mean(axis=1) * 1e6
sub_sem = (A_evkd_f[:,:,1]*1e6).std(axis=1) / np.sqrt(len(Subjects))

plt.plot(t_f,sub_mean, label='20')
plt.fill_between(t_f, sub_mean-sub_sem,sub_mean+sub_sem,alpha=0.5) 

plt.xlabel('Time (s)',fontsize=14)
plt.ylabel('\u03bcV',fontsize=14)
plt.tick_params(labelsize=12)
plt.legend(fontsize=12)

plt.savefig(os.path.join(fig_loc,'All_12vs20_noBaseline.svg'),format='svg')


#%% Plot Full time for individuals

plot_nums = 10 #len(Subjects)  
fig,ax = plt.subplots(int(np.ceil(plot_nums/2)),2,sharex=True,sharey=True,figsize=(14,12))
ax = np.reshape(ax,plot_nums)

for sub in range(plot_nums):
    cnd1 = 6
    cnd2 = 7
    
    mean_12 = A_epochs[sub][cnd1].mean(axis=0)
    sem_12 = A_epochs[sub][cnd1].std(axis=0) / np.sqrt(A_epochs[sub][cnd1].shape[0])
    
    ax[sub].plot(t_f,mean_12)
    ax[sub].fill_between(t_f,mean_12 - sem_12, mean_12+sem_12,alpha=0.5)
    
    mean_20= A_epochs[sub][cnd2].mean(axis=0)
    sem_20 = A_epochs[sub][cnd2].std(axis=0) / np.sqrt(A_epochs[sub][cnd2].shape[0])
    
    ax[sub].plot(t_f,mean_20)
    ax[sub].fill_between(t_f,mean_20-sem_20,mean_20+sem_20,alpha=0.5)

#%% Plot all Onsets, AB, or BA
plot_nums = 10 #len(Subjects)  
fig,ax = plt.subplots(int(np.ceil(plot_nums/2)),2,sharex=True,sharey=True,figsize=(14,12))
ax = np.reshape(ax,plot_nums)

cnd_plot = 1
labels = ['Onset', 'AtoB', 'BtoA']
conds_comp = [[0,1], [2,4], [3,5]]
   
fig.suptitle(labels[cnd_plot])

subs_to_plot = np.random.choice(range(0, 40), plot_nums, replace=False)


for sub in range(plot_nums):
    cnd1 = conds_comp[cnd_plot][0]
    cnd2 = conds_comp[cnd_plot][1]
    
    sub_p = subs_to_plot[sub]
    
    ax[sub].set_title(Subjects[sub_p]) 
    
    sub_norm = np.max(np.abs((A_epochs[sub_p][0].mean(axis=0) + A_epochs[sub_p][1].mean(axis=0)) / 2))
    sub_norm = 1 # get rid of normalization in plot for now
    
    onset12_mean = (A_epochs[sub_p][cnd1]).mean(axis=0) / sub_norm
    onset12_sem = (A_epochs[sub_p][cnd1]).std(axis=0) / np.sqrt(A_epochs[sub_p][cnd1].shape[0])
    
    ax[sub].plot(t,onset12_mean,label='12')  
    ax[sub].fill_between(t,onset12_mean - onset12_sem, onset12_mean + onset12_sem,alpha=0.5)
    
    onset20_mean = (A_epochs[sub_p][cnd2]).mean(axis=0) / sub_norm
    onset20_sem = (A_epochs[sub_p][cnd2]).std(axis=0) / np.sqrt(A_epochs[sub_p][cnd2].shape[0])
    
    ax[sub].plot(t,onset20_mean, label='20')
    ax[sub].fill_between(t,onset20_mean - onset20_sem, onset20_mean + onset20_sem,alpha=0.5)
    
    #ax[sub].plot(t,onset20_mean-onset12_mean,label='diff',color='k')
    
    #ax[sub].ticklabel_format(axis='y',style='sci',scilimits=(0,0))
    
    
ax[2].legend()
ax[plot_nums-1].set_xlabel('Time (sec)')
#ax[len(Subjects)-1].set_ylabel('Norm Amp')
    
plt.savefig(os.path.join(fig_loc, 'Asubj_' + labels[cnd_plot] + '.png'),format='png')

#%% Extract Features from EEG

#cond_bind = ['12 Onset', '20 Onset', '12AB', '12BA', '20AB', '20BA', '12 all','20 all']

feat_labels = ['Var12_norm', 'Var20_norm', 'mean12', 'mean20']
feat_condInds = [2, 4, 2, 4]
  
features = np.zeros([len(Subjects), len(feat_labels)])

t_0 = np.where(t>=0.000)[0][0]
t_2 = np.where(t>=0.300)[0][0]
t_3 = np.where(t>=0.800)[0][0]

for sub in range(len(Subjects)):
    
    features[sub,0] = A_evkd_cz[t_0:t_2,sub,feat_condInds[0]].var() / A_evkd_cz[t_0:t_2,sub,0].var() 
    features[sub,1] = A_evkd_cz[t_0:t_2,sub,feat_condInds[1]].var() / A_evkd_cz[t_0:t_2,sub,1].var() 
    features[sub,2] = A_evkd_cz[t_2:t_3,sub,feat_condInds[2]].mean() *1e6 #put in uV
    features[sub,3] = A_evkd_cz[t_2:t_3,sub,feat_condInds[3]].mean() *1e6 #put in uV
    
#%% Standardize features

features_zscale = StandardScaler().fit_transform(features)

    
#%% waveform pca for feature selection

# t_0 = np.where(t >= 0)[0][0]
# t_1 = np.where(t > 1)[0][0]

# O_resps = np.zeros((len(Subjects),(t_1-t_0)))
# B_resps = np.zeros((len(Subjects),(t_1-t_0)*2))
# A_resps = np.zeros((len(Subjects),(t_1-t_0)*2))
# AB_resps = np.zeros((len(Subjects),(t_1-t_0)*4))

# for sub in range(len(Subjects)):
#     Bresp = A_epochs[sub][2][:,t_0:t_1].mean(axis=0)
#     Bresp2 = A_epochs[sub][4][:,t_0:t_1].mean(axis=0)
    
#     Aresp = A_epochs[sub][3][:,t_0:t_1].mean(axis=0)
#     Aresp2 = A_epochs[sub][5][:,t_0:t_1].mean(axis=0)
    
#     On_resp = (A_epochs[sub][0][:,t_0:t_1].mean(axis=0) + A_epochs[sub][1][:,t_0:t_1].mean(axis=0) ) / 2 
    
#     O_resps[sub,:] = On_resp
#     A_resps[sub,:] = np.concatenate((Aresp,Aresp2))
#     B_resps[sub,:] = np.concatenate((Bresp,Bresp2))
#     AB_resps[sub,:] = np.concatenate((Aresp,Aresp2,Bresp,Bresp2))
    
    
# pca = PCA(n_components=3)

# #Aall_feature = pca.fit_transform(StandardScaler().fit_transform(A_resps))
# Aall_feature = pca.fit_transform(A_resps)
# Aall_expVar = pca.explained_variance_ratio_
# Aall_comp = pca.components_

# #Ball_feature = pca.fit_transform(StandardScaler().fit_transform(B_resps))  
# Ball_feature = pca.fit_transform(B_resps)
# Ball_expVar = pca.explained_variance_ratio_
# Ball_comp = pca.components_

# ABall_feature = pca.fit_transform(StandardScaler().fit_transform(AB_resps))
# ABall_expVar = pca.explained_variance_ratio_
# ABall_comp = pca.components_

# #O_feature = pca.fit_transform(StandardScaler().fit_transform(O_resps))
# O_feature = pca.fit_transform(O_resps)
# O_expVar = pca.explained_variance_ratio_
# O_comp = pca.components_



# # plt.figure()
# # plt.plot(np.concatenate((t[t_0:t_1],t[t_0:t_1]+1+1/4096)),B_resps.mean(axis=0))
# # plt.plot(np.concatenate((t[t_0:t_1],t[t_0:t_1]+1+1/4096)),Ball_comp[0,:], label ='Coherent', color='Black')


# plt.figure()
# plt.scatter(O_feature[:,0],Ball_feature[:,0])

# plt.figure()
# plt.scatter(O_feature[:,0],Aall_feature[:,0])

# plt.figure()
# #plt.plot(t[t_0:t_1],O_comp[0,:],label='Onset',linewidth='2')
# plt.plot(np.concatenate((t[t_0:t_1],t[t_0:t_1]+1+1/4096)),Aall_comp[0,:], label= 'Incoherent', color='Grey')
# plt.plot(np.concatenate((t[t_0:t_1],t[t_0:t_1]+1+1/4096)),Ball_comp[0,:], label ='Coherent', color='Black')
# plt.xticks([0,0.5,1.0, 1.5, 2.0])
# plt.yticks([0, 0.01, .02])
# plt.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
# plt.legend(loc=2)
# plt.ylabel('PCA Feature Weight')
# plt.xlabel('Time (sec)')
# #plt.title('EEG Feature From PCA')

# plt.savefig(os.path.join(fig_loc,'PCA_FeatWeights.svg'),format='svg')

# fig = plt.figure()
# #fig.set_size_inches(12,10)
# plt.rcParams.update({'font.size': 12})
# #plt.plot(t[t_0:t_1],O_comp[0,:],label='Onset',linewidth='2')
# plt.plot(t[t_0:t_1], Aall_comp[0,4097:], label= 'Incoherent',linewidth='2')
# plt.plot(t[t_0:t_1], -Ball_comp[0,4097:], label ='Coherent',linewidth='2')
# plt.legend(loc = 4)
# plt.ylabel('Weight')
# plt.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
# plt.xlabel('Time (s)')
# plt.title('EEG Feature Weights From PCA')
# plt.xticks([0,0.5,1.0])
# #plt.yticks([0, 0.02, 0.04])
# plt.yticks([0, 0.01, .02])


# #plt.plot(np.concatenate((t[t_0:t_1],t[t_0:t_1]+1+1/4096, t[t_0:t_1]+2+1/4096,t[t_0:t_1]+3+1/4096 )),
#  #        ABall_comp[0,:])
 
# plt.figure()
# plt.scatter(Aall_feature[:,0],Aall_feature[:,0]/ O_feature[:,0])

# ABall_feature = ABall_feature[:,0:1] #/ O_feature[:,0]
# Aall_feature = Aall_feature[:,0:1] #/ O_feature[:,0]
# Ball_feature = Ball_feature[:,0:1] #/ O_feature[:,0],
# O_feature = O_feature[:,0:1]
    
#%% Explore features
    
plt.figure()
plt.scatter(features[:,0],features[:,1], label=feat_labels[1])
# plt.scatter(features[:,0],features[:,2], label=feat_labels[2])
# plt.scatter(features[:,0],features[:,3], label=feat_labels[3])
plt.xlabel(feat_labels[0])
plt.legend()

plt.figure()
plt.scatter(features[:,2],features[:,3], label=feat_labels[2])
plt.xlabel(feat_labels[2])
plt.legend()

plt.figure()
plt.scatter(features[:,1], features[:,3])
plt.xlabel(feat_labels[1])
plt.ylabel(feat_labels[3])



#%% PCA on feature 

pca = PCA(n_components=2)
pca_on = pca.fit_transform(features_zscale)
pca_on_expVar = pca.explained_variance_ratio_

plt.figure()
plt.plot(pca.components_.T)
plt.title('On PCA')

pca_feats = pca_on



#%% Add B pca feature and full waveform pca feature

# feat_labels.append('PCA_B')
# features = np.append(features,pca_B[:,0:1],axis=1)

# feat_labels.append('PCA_Ball')
# Ball_feature = StandardScaler().fit_transform(Ball_feature)
# features = np.append(features,Ball_feature,axis=1)


#%% Use PCA to condense temporal Features from full time waveform

# t1 = np.where(t>=0)[0][0]
# t2 = np.where(t>=1)[0][0]
# tp = t[t1:t2]

# #Evkd_all = np.zeros((tp.size*6,len(Subjects)))
# Evkd_conds = np.zeros((tp.size,len(Subjects),len(conds_save)))

# for sub in range(len(Subjects)):
#     A_evkd_ = np.array([])
#     for cond in range(len(conds_save)):
#         evkd = A_epochs[sub][cond].mean(axis=0)[t1:t2]
#         #evkd = signal.decimate(evkd,16) --- computer can handle but okay to do this if want to
#         A_evkd_ = np.concatenate((A_evkd_,evkd))
#         Evkd_conds[:,sub,cond] = evkd
    
#     #Evkd_all[:,sub] = A_evkd_
    
# #Condense two onset conditions into 1    
# Evkd_conds[:,:,0] = (Evkd_conds[:,:,0] + Evkd_conds[:,:,1]) / 2
# Evkd_conds = np.delete(Evkd_conds,1,axis=2)

# conds_s = ['Onset', '12AB', '12BA', '20AB', '20BA']

# fix,ax = plt.subplots(3,2)
# ax = np.reshape(ax,len(conds_save))
# for cond in range(len(conds_s)):
#     ax[cond].plot(tp,Evkd_conds[:,:,cond].mean(axis=1))


# t1 = np.where(tp>=0)[0][0]
# t2 = np.where(tp>=0.4)[0][0]

# pca = PCA(n_components=10)
# X_pca = []
# X_expVar = []
# X_comp = []

# for cond in range(len(conds_s)):
#     X = pca.fit_transform(Evkd_conds[t1:t2,:,cond].T)
#     X_pca.append(X)
#     X_expVar.append(pca.explained_variance_ratio_)
#     X_comp.append(pca.components_)
    
    
# for cond in range(len(conds_s)):
#     print("Varaince explained in cond " + str(cond) + ": " + str(X_expVar[cond][0:2].sum()))
    

# fix,ax = plt.subplots(3,2)
# ax = np.reshape(ax,6)
# for cond in range(len(conds_s)):
#     ax[cond].plot(tp[t1:t2],X_comp[cond][0:features_matter2,:].T)
#     ax[cond].plot(tp[t1:t2],np.sum(X_comp[cond][0:2,:],axis=0),color='k')
    
    
        
# fix,ax = plt.subplots(3,2)
# ax = np.reshape(ax,6)
# for cond in range(len(conds_s)):
#     ax[cond].scatter(X_pca[0][:,0:3].sum(axis=1), X_pca[cond][:,0:3].sum(axis=1))


# fig = plt.figure()
# ax = Axes3D(fig)
# ax.scatter(X_pca[0][:,0], X_pca[0][:,1], X_pca[0][:,2])


# np.append(features,pca_B[:,0:1],axis=1)
#%% Load Binding Behavior

data_loc_bindingBeh = '/media/ravinderjit/Data_Drive/Data/MTB_Behavior/MTB_beh/'
beh_bind = sio.loadmat(data_loc_bindingBeh + 'BindingBeh.mat',squeeze_me=True)


Subj_bind = list(beh_bind['Subjects'])
acc_bind = beh_bind['acc']

#Sort behavioral data to be in same order as physio data
bind_sort = np.zeros(len(Subjects),dtype=int)
for s in range(len(Subjects)):
    bind_sort[s] = Subj_bind.index(Subjects[s])
    
    
acc_bind = acc_bind[:,bind_sort]
# 0 = 19:20
# 1 = 17-20
# 2 = 15-20
# 3 = 13:20
# 4 = 1, 8, 14, 20
# 5 = 1, 4, 8, 12, 16, 20
# 6 = 1, 4, 6, 9, 12, 15, 17, 20

thresh_coh = acc_bind[1,:]
consec_coh = acc_bind[2:4,:].mean(axis=0)
#spaced_coh = acc_bind[5:7,:].mean(axis=0)
spaced_coh = acc_bind[6,:]


beh_conds = ['2','4','6','8', '4 spaced','6 spaced','8 spaced']


#%% Group differences

#young vs old

plt.figure()
plt.plot(np.arange(7),acc_bind[:,young].mean(axis=1),label='young')
plt.fill_between(np.arange(7),acc_bind[:,young].mean(axis=1) - acc_bind[:,young].std(axis=1) /np.sqrt(np.sum(young)),
                 acc_bind[:,young].mean(axis=1) + acc_bind[:,young].std(axis=1) /np.sqrt(np.sum(young)),alpha=0.5)
plt.plot(np.arange(7), acc_bind[:,old].mean(axis=1),label='old')
plt.fill_between(np.arange(7),acc_bind[:,old].mean(axis=1) - acc_bind[:,old].std(axis=1) /np.sqrt(np.sum(old)),
                 acc_bind[:,old].mean(axis=1) + acc_bind[:,old].std(axis=1) /np.sqrt(np.sum(old)),alpha=0.5)
plt.legend()
plt.xticks(ticks=np.arange(7),labels=beh_conds)
plt.xlabel('Behavior condition')



#%% Plot good vs bad performers

cond_bind = ['12 Onset', '20 Onset', '12AB', '12BA', '20AB', '20BA', '12 all','20 all']
        
conds_comp = [[0,1], [2,4], [3,5]]
beh_c = 1
labels = ['Onset', 'Incoherent to Coherent', 'Coherent to Incoherent']
    
fig,ax = plt.subplots(3,1,sharex=True)

#beh = acc_bind[1:3,:].mean(axis=0)
beh = acc_bind[beh_c,:]
good = beh >= np.percentile(beh,50)
bad = beh < np.percentile(beh,50)

# beh = acc_bind[4:5,:].mean(axis=0)
# # good = beh >= np.median(beh)
# # bad = beh < np.median(beh)
# good = beh > np.percentile(beh,75)
# bad = beh < np.percentile(beh,25)


print('good: ' + str(np.sum(good)) + ' bad: ' + str(np.sum(bad)) + '\n' + 
      'good: ' + str(np.round(np.percentile(beh,75)*100)) + '%' + 
      ' bad: ' + str(np.round(np.percentile(beh,25)*100)) + '%' )

conds_comp = [[1,1], [4,4], [5,5]]
labels = ['Onset', 'Incoherent to Coherent 20', 'Coherent to Incoherent 20']
# conds_comp = [[0,0], [2,2], [3,3]]
# labels = ['Onset', 'Incoherent to Coherent 12', 'Coherent to Incoherent 12']


for jj in range(3):
    cnd1 = conds_comp[jj][0]
    cnd2 = conds_comp[jj][1]
    
    onset12_mean = A_evkd_cz[:,good,cnd1].mean(axis=1)
    onset12_sem = A_evkd_cz[:,good,cnd1].std(axis=1) / np.sqrt(A_evkd_cz[:,good,cnd1].shape[1])
    
    ax[jj].plot(t,onset12_mean,label='Good',color='forestgreen')  
    ax[jj].fill_between(t,onset12_mean - onset12_sem, onset12_mean + onset12_sem,alpha=0.5, color='forestgreen')
    
    onset20_mean = A_evkd_cz[:,bad,cnd2].mean(axis=1)
    onset20_sem = A_evkd_cz[:,bad,cnd2].std(axis=1) / np.sqrt(A_evkd_cz[:,bad,cnd2].shape[1])
    
    ax[jj].plot(t,onset20_mean,label='Bad', color='indianred')  
    ax[jj].fill_between(t,onset20_mean - onset20_sem, onset20_mean + onset20_sem,alpha=0.5,color='indianred')
    
    ax[jj].ticklabel_format(axis='y',style='sci',scilimits=(0,0))
    ax[jj].set_title(labels[jj])


ax[0].legend()
ax[2].set_xlabel('Time')
#ax[2].set_ylabel('$\mu$V')
fig.suptitle('Beh Cond:'  + beh_conds[beh_c])

#plt.savefig(os.path.join(fig_loc,'GoodvsBad_SpacedCoh_20.png'),format='png')

#%% Load Jane
data_loc_jane = '/media/ravinderjit/Data_Drive/Data/MTB_Behavior/SIN_Info_JANE/'

jane = sio.loadmat(data_loc_jane + 'SINinfo_Jane.mat',squeeze_me=True)

Subj_jane = list(jane['Subjects'])
thresh_jane = jane['thresholds']

index_jane = sortSubs(Subj_jane, Subjects)
thresh_jane = thresh_jane[index_jane]


#%% Load MRT

data_loc_mrt = '/media/ravinderjit/Data_Drive/Stim_Analysis/Analysis/SnapLabOnline/MTB_MRT_online/'

mrt = sio.loadmat(data_loc_mrt + 'MTB_MRT.mat', squeeze_me=True)


Subj_mrt = list(mrt['Subjects'])
thresh_mrt = mrt['thresholds']
lapse_mrt = mrt['lapse']

index_mrt = sortSubs_exception(Subj_mrt, Subjects)

thresh_mrt_temp = np.zeros(42)
lapse_mrt_temp = np.zeros(42)
for k in range(42):
    if index_mrt[k] >=0:
        thresh_mrt_temp[k] = thresh_mrt[index_mrt[k]]
        lapse_mrt_temp[k] = lapse_mrt[index_mrt[k]]
    else:
        thresh_mrt_temp[k] = -1000 #dummy value
        lapse_mrt_temp[k] = -1000 #dummy value
        
thresh_mrt = thresh_mrt_temp
lapse_mrt = lapse_mrt_temp


#%% Load CMR

data_loc_cmr = '/media/ravinderjit/Data_Drive/Data/MTB_Behavior/CMR_beh/'

cmr = sio.loadmat(data_loc_cmr + 'CMRclickMod.mat',squeeze_me=True)

Subj_cmr = list(cmr['Subjects'])
thresh_cmr = cmr['CMR']
lapse_cmr = cmr['lapse']
cohthresh_cmr = -cmr['coh_thresh']

index_cmr = sortSubs(Subj_cmr, Subjects)
thresh_cmr = thresh_cmr[index_cmr]
lapse_cmr = lapse_cmr[index_cmr]
cohthresh_cmr = cohthresh_cmr[index_cmr]


#%% Plot good vs bad performers Jane

#cond_bind = ['12 Onset', '20 Onset', '12AB', '12BA', '20AB', '20BA', '12 all','20 all']

labels = ['Onset', 'Incoherent to Coherent', 'Coherent to Incoherent']
    
fig,ax = plt.subplots(3,1,sharex=True)

#beh = acc_bind[1:3,:].mean(axis=0)
beh = thresh_jane
good = beh <= np.percentile(beh,50)
bad = beh > np.percentile(beh,50)

print('good: ' + str(np.sum(good)) + ' bad: ' + str(np.sum(bad)) + '\n' + 
      'bad: ' + str(np.round(np.percentile(beh,50)))  + 
      ' good: ' + str(np.round(np.percentile(beh,50))) )

conds_comp = [[1,1], [4,4], [5,5]]
labels = ['Onset', 'Incoherent to Coherent 20', 'Coherent to Incoherent 20']
# conds_comp = [[0,0], [2,2], [3,3]]
# labels = ['Onset', 'Incoherent to Coherent 12', 'Coherent to Incoherent 12']

for jj in range(3):
    cnd1 = conds_comp[jj][0]
    cnd2 = conds_comp[jj][1]
    
    onset12_mean = A_evkd_cz[:,good,cnd1].mean(axis=1)
    onset12_sem = A_evkd_cz[:,good,cnd1].std(axis=1) / np.sqrt(A_evkd_cz[:,good,cnd1].shape[1])
    
    ax[jj].plot(t,onset12_mean,label='Good',color='forestgreen')  
    ax[jj].fill_between(t,onset12_mean - onset12_sem, onset12_mean + onset12_sem,alpha=0.5, color='forestgreen')
    
    onset20_mean = A_evkd_cz[:,bad,cnd2].mean(axis=1)
    onset20_sem = A_evkd_cz[:,bad,cnd2].std(axis=1) / np.sqrt(A_evkd_cz[:,bad,cnd2].shape[1])
    
    ax[jj].plot(t,onset20_mean,label='Bad', color='indianred')  
    ax[jj].fill_between(t,onset20_mean - onset20_sem, onset20_mean + onset20_sem,alpha=0.5,color='indianred')
    
    ax[jj].ticklabel_format(axis='y',style='sci',scilimits=(0,0))
    ax[jj].set_title(labels[jj])


ax[0].legend()
ax[2].set_xlabel('Time')
#ax[2].set_ylabel('$\mu$V')
fig.suptitle('Jane')

#%% Plot good vs bad performers MRT

labels = ['Onset', 'Incoherent to Coherent', 'Coherent to Incoherent']
    
fig,ax = plt.subplots(3,1,sharex=True)

beh = thresh_mrt
good = np.logical_and(beh <= np.percentile(beh,50), beh > -100) #logical and to deal with exception that not all participants did MRT
bad = np.logical_and(beh > np.percentile(beh,50), beh > -100)

print('good: ' + str(np.sum(good)) + ' bad: ' + str(np.sum(bad)) + '\n' + 
      'bad: ' + str(np.round(np.percentile(beh,75)))  + 
      ' good: ' + str(np.round(np.percentile(beh,25))) )

conds_comp = [[1,1], [4,4], [5,5]]
labels = ['Onset', 'Incoherent to Coherent 20', 'Coherent to Incoherent 20']
conds_comp = [[0,0], [2,2], [3,3]]
labels = ['Onset', 'Incoherent to Coherent 12', 'Coherent to Incoherent 12']

for jj in range(3):
    cnd1 = conds_comp[jj][0]
    cnd2 = conds_comp[jj][1]
    
    onset12_mean = A_evkd_cz[:,good,cnd1].mean(axis=1)
    onset12_sem = A_evkd_cz[:,good,cnd1].std(axis=1) / np.sqrt(A_evkd_cz[:,good,cnd1].shape[1])
    
    ax[jj].plot(t,onset12_mean,label='Good',color='forestgreen')  
    ax[jj].fill_between(t,onset12_mean - onset12_sem, onset12_mean + onset12_sem,alpha=0.5, color='forestgreen')
    
    onset20_mean = A_evkd_cz[:,bad,cnd2].mean(axis=1)
    onset20_sem = A_evkd_cz[:,bad,cnd2].std(axis=1) / np.sqrt(A_evkd_cz[:,bad,cnd2].shape[1])
    
    ax[jj].plot(t,onset20_mean,label='Bad', color='indianred')  
    ax[jj].fill_between(t,onset20_mean - onset20_sem, onset20_mean + onset20_sem,alpha=0.5,color='indianred')
    
    ax[jj].ticklabel_format(axis='y',style='sci',scilimits=(0,0))
    ax[jj].set_title(labels[jj])


ax[0].legend()
ax[2].set_xlabel('Time')
#ax[2].set_ylabel('$\mu$V')
fig.suptitle('MRT')

#%% Plot good vs bad performers CMR

labels = ['Onset', 'Incoherent to Coherent', 'Coherent to Incoherent']
    
fig,ax = plt.subplots(3,1,sharex=True)

#beh = acc_bind[1:3,:].mean(axis=0)
beh = thresh_cmr
good = beh >= np.percentile(beh,50)
bad = beh < np.percentile(beh,50)

print('good: ' + str(np.sum(good)) + ' bad: ' + str(np.sum(bad)) + '\n' + 
      'bad: ' + str(np.round(np.percentile(beh,25)))  + 
      ' good: ' + str(np.round(np.percentile(beh,75))) )

conds_comp = [[1,1], [4,4], [5,5]]
labels = ['Onset', 'Incoherent to Coherent 20', 'Coherent to Incoherent 20']
# conds_comp = [[0,0], [2,2], [3,3]]
# labels = ['Onset', 'Incoherent to Coherent 12', 'Coherent to Incoherent 12']

for jj in range(3):
    cnd1 = conds_comp[jj][0]
    cnd2 = conds_comp[jj][1]
    
    onset12_mean = A_evkd_cz[:,good,cnd1].mean(axis=1)
    onset12_sem = A_evkd_cz[:,good,cnd1].std(axis=1) / np.sqrt(A_evkd_cz[:,good,cnd1].shape[1])
    
    ax[jj].plot(t,onset12_mean,label='Good',color='forestgreen')  
    ax[jj].fill_between(t,onset12_mean - onset12_sem, onset12_mean + onset12_sem,alpha=0.5, color='forestgreen')
    
    onset20_mean = A_evkd_cz[:,bad,cnd2].mean(axis=1)
    onset20_sem = A_evkd_cz[:,bad,cnd2].std(axis=1) / np.sqrt(A_evkd_cz[:,bad,cnd2].shape[1])
    
    ax[jj].plot(t,onset20_mean,label='Bad', color='indianred')  
    ax[jj].fill_between(t,onset20_mean - onset20_sem, onset20_mean + onset20_sem,alpha=0.5,color='indianred')
    
    ax[jj].ticklabel_format(axis='y',style='sci',scilimits=(0,0))
    ax[jj].set_title(labels[jj])


ax[0].legend()
ax[2].set_xlabel('Time')
#ax[2].set_ylabel('$\mu$V')
fig.suptitle('CMR')

#%% Plot good vs bad performers CMR cohthresh

labels = ['Onset', 'Incoherent to Coherent', 'Coherent to Incoherent']
    
fig,ax = plt.subplots(3,1,sharex=True)

#beh = acc_bind[1:3,:].mean(axis=0)
beh = cohthresh_cmr
good = beh >= np.percentile(beh,50)
bad = beh < np.percentile(beh,50)

print('good: ' + str(np.sum(good)) + ' bad: ' + str(np.sum(bad)) + '\n' + 
      'bad: ' + str(np.round(np.percentile(beh,25)))  + 
      ' good: ' + str(np.round(np.percentile(beh,75))) )

conds_comp = [[1,1], [4,4], [5,5]]
labels = ['Onset', 'Incoherent to Coherent 20', 'Coherent to Incoherent 20']
# conds_comp = [[0,0], [2,2], [3,3]]
# labels = ['Onset', 'Incoherent to Coherent 12', 'Coherent to Incoherent 12']

for jj in range(3):
    cnd1 = conds_comp[jj][0]
    cnd2 = conds_comp[jj][1]
    
    onset12_mean = A_evkd_cz[:,good,cnd1].mean(axis=1)
    onset12_sem = A_evkd_cz[:,good,cnd1].std(axis=1) / np.sqrt(A_evkd_cz[:,good,cnd1].shape[1])
    
    ax[jj].plot(t,onset12_mean,label='Good',color='forestgreen')  
    ax[jj].fill_between(t,onset12_mean - onset12_sem, onset12_mean + onset12_sem,alpha=0.5, color='forestgreen')
    
    onset20_mean = A_evkd_cz[:,bad,cnd2].mean(axis=1)
    onset20_sem = A_evkd_cz[:,bad,cnd2].std(axis=1) / np.sqrt(A_evkd_cz[:,bad,cnd2].shape[1])
    
    ax[jj].plot(t,onset20_mean,label='Bad', color='indianred')  
    ax[jj].fill_between(t,onset20_mean - onset20_sem, onset20_mean + onset20_sem,alpha=0.5,color='indianred')
    
    ax[jj].ticklabel_format(axis='y',style='sci',scilimits=(0,0))
    ax[jj].set_title(labels[jj])


ax[0].legend()
ax[2].set_xlabel('Time')
#ax[2].set_ylabel('$\mu$V')
fig.suptitle('CMR coThresh')

#%% Plot features for various conditions

#Jane
beh = thresh_jane
good = beh <= np.percentile(beh,50)
bad = beh > np.percentile(beh,50)

plt.figure()
gbox = plt.boxplot(features[good,:], positions = [1,2,3,4], widths=0.33*np.ones(4),patch_artist=True)
bbox = plt.boxplot(features[bad,:], positions = np.array([1,2,3,4]) + 0.5, widths=0.33*np.ones(4), patch_artist=True)

for el in gbox['boxes']:
    el.set(color='green')
for el in bbox['boxes']:
    el.set(color='red')
    
plt.title('Jane')
plt.xticks([1.25, 2.25, 3.25, 4.25], feat_labels)

#MRT
beh = thresh_mrt
good = np.logical_and(beh <= np.percentile(beh,50), beh > -100) #logical and to deal with exception that not all participants did MRT
bad = np.logical_and(beh > np.percentile(beh,50), beh > -100)

plt.figure()
gbox = plt.boxplot(features[good,:], positions = [1,2,3,4], widths=0.33*np.ones(4),patch_artist=True)
bbox = plt.boxplot(features[bad,:], positions = np.array([1,2,3,4]) + 0.5, widths=0.33*np.ones(4), patch_artist=True)

for el in gbox['boxes']:
    el.set(color='green')
for el in bbox['boxes']:
    el.set(color='red')
    
plt.title('MRT')
plt.xticks([1.25, 2.25, 3.25, 4.25], feat_labels)


#CMR

beh = thresh_cmr
good = beh >= np.percentile(beh,50)
bad = beh < np.percentile(beh,50)

plt.figure()
gbox = plt.boxplot(features[good,:], positions = [1,2,3,4], widths=0.33*np.ones(4),patch_artist=True)
bbox = plt.boxplot(features[bad,:], positions = np.array([1,2,3,4]) + 0.5, widths=0.33*np.ones(4), patch_artist=True)

for el in gbox['boxes']:
    el.set(color='green')
for el in bbox['boxes']:
    el.set(color='red')
    
plt.title('CMR')
plt.xticks([1.25, 2.25, 3.25, 4.25], feat_labels)

#Binding Beh
beh_c = 2
beh = acc_bind[beh_c,:]
good = beh >= np.percentile(beh,50)
bad = beh < np.percentile(beh,50)

plt.figure()
gbox = plt.boxplot(features[good,:], positions = [1,2,3,4], widths=0.33*np.ones(4),patch_artist=True)
bbox = plt.boxplot(features[bad,:], positions = np.array([1,2,3,4]) + 0.5, widths=0.33*np.ones(4), patch_artist=True)

for el in gbox['boxes']:
    el.set(color='green')
for el in bbox['boxes']:
    el.set(color='red')
    
plt.title('Beh Cond:'  + beh_conds[beh_c])
plt.xticks([1.25, 2.25, 3.25, 4.25], feat_labels)

#Young and Old
good = young
bad = old

plt.figure()
gbox = plt.boxplot(features[good,:], positions = [1,2,3,4], widths=0.33*np.ones(4),patch_artist=True)
bbox = plt.boxplot(features[bad,:], positions = np.array([1,2,3,4]) + 0.5, widths=0.33*np.ones(4), patch_artist=True)

for el in gbox['boxes']:
    el.set(color='green')
for el in bbox['boxes']:
    el.set(color='red')
    
plt.title('age')
plt.xticks([1.25, 2.25, 3.25, 4.25], feat_labels)

#%% Now with PCA features

#Jane
beh = thresh_jane
good = beh <= np.percentile(beh,50)
bad = beh > np.percentile(beh,50)

plt.figure()
gbox = plt.boxplot(pca_feats[good,:], positions = [1,2], widths=0.33,patch_artist=True)
bbox = plt.boxplot(pca_feats[bad,:], positions = np.array([1,2]) + 0.5, widths=0.33, patch_artist=True)

for el in gbox['boxes']:
    el.set(color='green')
for el in bbox['boxes']:
    el.set(color='red')
    
plt.title('Jane')
plt.xticks([1.25, 2.25], ['PCA1', 'PCA2'])

#MRT
beh = thresh_mrt
good = np.logical_and(beh <= np.percentile(beh,50), beh > -100) #logical and to deal with exception that not all participants did MRT
bad = np.logical_and(beh > np.percentile(beh,50), beh > -100)

plt.figure()
gbox = plt.boxplot(pca_feats[good,:], positions = [1,2], widths=0.33,patch_artist=True)
bbox = plt.boxplot(pca_feats[bad,:], positions = np.array([1,2]) + 0.5, widths=0.33, patch_artist=True)

for el in gbox['boxes']:
    el.set(color='green')
for el in bbox['boxes']:
    el.set(color='red')
    
plt.title('MRT')
plt.xticks([1.25, 2.25], ['PCA1', 'PCA2'])


#CMR

beh = thresh_cmr
good = beh <= np.percentile(beh,50)
bad = beh > np.percentile(beh,50)

plt.figure()
gbox = plt.boxplot(pca_feats[good,:], positions = [1,2], widths=0.33,patch_artist=True)
bbox = plt.boxplot(pca_feats[bad,:], positions = np.array([1,2]) + 0.5, widths=0.33, patch_artist=True)

for el in gbox['boxes']:
    el.set(color='green')
for el in bbox['boxes']:
    el.set(color='red')
    
plt.title('CMR')
plt.xticks([1.25, 2.25], ['PCA1', 'PCA2'])

#Binding Beh
beh = acc_bind[1,:]
good = beh >= np.percentile(beh,50)
bad = beh < np.percentile(beh,50)

plt.figure()
gbox = plt.boxplot(pca_feats[good,:], positions = [1,2], widths=0.33,patch_artist=True)
bbox = plt.boxplot(pca_feats[bad,:], positions = np.array([1,2]) + 0.5, widths=0.33, patch_artist=True)

for el in gbox['boxes']:
    el.set(color='green')
for el in bbox['boxes']:
    el.set(color='red')
    
plt.title('Beh Cond:'  + beh_conds[beh_c])
plt.xticks([1.25, 2.25], ['PCA1', 'PCA2'])

#Young and Old
good = young
bad = old

plt.figure()
gbox = plt.boxplot(pca_feats[good,:], positions = [1,2], widths=0.33,patch_artist=True)
bbox = plt.boxplot(pca_feats[bad,:], positions = np.array([1,2]) + 0.5, widths=0.33, patch_artist=True)

for el in gbox['boxes']:
    el.set(color='green')
for el in bbox['boxes']:
    el.set(color='red')
    
plt.title('age')
plt.xticks([1.25, 2.25], ['PCA1', 'PCA2'])

#%% Look at just average of 12 and 20 mean
feat_ss2012 = features[:,2:].mean(axis=1)


#Binding, Jane, MRT, CMR, Age

beh_4bind = acc_bind[1,:]
good_4bind = beh_4bind >= np.percentile(beh_4bind,50)
bad_4bind = beh_4bind < np.percentile(beh_4bind,50)

beh_6bind = acc_bind[2,:]
good_6bind = beh_6bind >= np.percentile(beh_6bind,50)
bad_6bind = beh_6bind < np.percentile(beh_6bind,50)

plt.figure()
gbox = plt.boxplot(feat_ss2012[good_4bind], positions=[1],widths=[0.33],patch_artist=True)
bbox = plt.boxplot(feat_ss2012[bad_4bind], positions=[1.5], widths=[0.33], patch_artist=True)

gbox['boxes'][0].set(color='green')
bbox['boxes'][0].set(color='red')

[t_v, p] = stats.ttest_ind(feat_ss2012[good_4bind], feat_ss2012[bad_4bind])
print('For Bind 4: P-value is: ' +str(p/2))

gbox = plt.boxplot(feat_ss2012[good_6bind], positions=[2],widths=[0.33],patch_artist=True)
bbox = plt.boxplot(feat_ss2012[bad_6bind], positions=[2.5], widths=[0.33], patch_artist=True)

gbox['boxes'][0].set(color='green')
bbox['boxes'][0].set(color='red')

[t_v, p] = stats.ttest_ind(feat_ss2012[good_6bind], feat_ss2012[bad_6bind])
print('For Bind 6: P-value is: ' +str(p/2))

#Jane
beh = thresh_jane
good_jane = beh <= np.percentile(beh,50)
bad_jane = beh > np.percentile(beh,50)

gbox = plt.boxplot(feat_ss2012[good_jane], positions=[3],widths=[0.33],patch_artist=True)
bbox = plt.boxplot(feat_ss2012[bad_jane], positions=[3.5], widths=[0.33], patch_artist=True)
gbox['boxes'][0].set(color='green')
bbox['boxes'][0].set(color='red')

[t_v, p] = stats.ttest_ind(feat_ss2012[good_jane], feat_ss2012[bad_jane])
print('Jane P-value is: ' +str(p/2))

#MRT
beh = thresh_mrt
good_mrt = np.logical_and(beh <= np.percentile(beh,50), beh > -100) #logical and to deal with exception that not all participants did MRT
bad_mrt = np.logical_and(beh > np.percentile(beh,50), beh > -100)

gbox = plt.boxplot(feat_ss2012[good_mrt], positions=[4],widths=[0.33],patch_artist=True)
bbox = plt.boxplot(feat_ss2012[bad_mrt], positions=[4.5], widths=[0.33], patch_artist=True)
gbox['boxes'][0].set(color='green')
bbox['boxes'][0].set(color='red')

[t_v, p] = stats.ttest_ind(feat_ss2012[good_mrt], feat_ss2012[bad_mrt])
print('MRT P-value is: ' +str(p/2))

#CMR
beh = thresh_cmr
good_cmr = beh >= np.percentile(beh,50)
bad_cmr = beh < np.percentile(beh,50)

gbox = plt.boxplot(feat_ss2012[good_cmr], positions=[5],widths=[0.33],patch_artist=True)
bbox = plt.boxplot(feat_ss2012[bad_cmr], positions=[5.5], widths=[0.33], patch_artist=True)
gbox['boxes'][0].set(color='green')
bbox['boxes'][0].set(color='red')

[t_v, p] = stats.ttest_ind(feat_ss2012[good_cmr], feat_ss2012[bad_cmr])
print('CMR P-value is: ' +str(p/2))

#Age

gbox = plt.boxplot(feat_ss2012[young], positions = [6], widths=[0.33],patch_artist=True)
bbox = plt.boxplot(feat_ss2012[old], positions = [6.5], widths=[0.33], patch_artist=True)
gbox['boxes'][0].set(color='green')
bbox['boxes'][0].set(color='red')

[t_v, p] = stats.ttest_ind(feat_ss2012[young], feat_ss2012[old])
print('Age P-value is: ' +str(p/2))


plt.xticks([1.25,2.25,3.25,4.25,5.25,6.25],labels=['Bind 4', 'Bind 6', 'Jane', 'MRT', 'CMR', 'Age'])



#%% MRT Stats

feat_added = feat_ss2012#(-features[:,2] + -features[:,3]) / 2

mrt_mask = thresh_mrt > -100
y = thresh_mrt[mrt_mask]
X = feat_added[mrt_mask]
X = np.vstack((X,lapse_mrt[mrt_mask])).T
s
X = sm.add_constant(X)

results = sm.OLS(y,X).fit()
print(results.summary())

plt.figure()
plt.scatter(feat_added[mrt_mask], y)

#%% CMR stats
feat_added = (-features[:,2] + -features[:,3]) / 2

y = thresh_cmr
X = np.vstack((feat_added,lapse_cmr)).T


X = sm.add_constant(X)

results = sm.OLS(y,X).fit()
print(results.summary())

plt.figure()
plt.scatter(feat_added, y)
plt.title('CMR')

#%% CMR cothresh

feat_added = (-features[:,2] + -features[:,3]) / 2

y = cohthresh_cmr
X = np.vstack((feat_added,lapse_cmr)).T


X = sm.add_constant(X)

results = sm.OLS(y,X).fit()
print(results.summary())

plt.figure()
plt.scatter(feat_added, y)
plt.title('CMR')



#%% Jane Stats
feat_added = (-features[:,2] + -features[:,3]) / 2

y = thresh_jane
X = feat_added

X = sm.add_constant(X)

results = sm.OLS(y,X).fit()
print(results.summary())

plt.figure()
plt.scatter(X[:,1],y)
plt.title('Jane')

#%% Jane Vs MRT

plt.figure()
plt.scatter(thresh_jane[mrt_mask], thresh_mrt[mrt_mask])

y = thresh_jane[mrt_mask]
X = thresh_mrt[mrt_mask]

X = sm.add_constant(X)

results = sm.OLS(y,X).fit()
print(results.summary())

#%% CMR vs Jane

plt.figure()
plt.scatter(thresh_cmr, thresh_jane)

y = thresh_jane
X = thresh_cmr

X = sm.add_constant(X)

results = sm.OLS(y,X).fit()
print(results.summary())

#%% CMR Vs MRT

plt.figure()
plt.scatter(thresh_cmr[mrt_mask], thresh_mrt[mrt_mask])

y = thresh_mrt[mrt_mask]
X = thresh_cmr[mrt_mask]

X = sm.add_constant(X)

results = sm.OLS(y,X).fit()
print(results.summary())



#%% Save stuff

# save_dict = {'threshCoh': thresh_coh, 'spacedCoh': spaced_coh,
#              'consecCoh': consec_coh, 'feats_Bon': pca_feats_Bon,
#              'feats_Bmn': pca_feats_Bmn, 'feats_B': pca_feats_B,
#              'B_Weights': pca_weights_B, 'B_expVar': pca_expVar_B, 
#              'Bon_Weights': pca_weights_Bon, 'Bmn_Weights': pca_weights_Bmn, 
#              'Bon_expVar': pca_expVar_Bon, 'Bmn_weights': pca_weights_Bmn,
#              'Aall_feat': Aall_feature, 'Ball_feat': Ball_feature,
#              'Aall_comp': Aall_comp, 'Ball_comp': Ball_comp,
#              'Aall_expVar':Aall_expVar, 'Ball_expVar': Ball_expVar,
#              'O_comp': O_comp, 'O_expVar': O_expVar,'O_feature':O_feature,
#              'Subjects': Subjects}

save_dict = {'threshCoh': thresh_coh, 'spacedCoh': spaced_coh,
             'consecCoh': consec_coh, 'Subjects': Subjects, 'age': age,
             'features': features, 'feat_labels': feat_labels}

mod_vars_loc = '/media/ravinderjit/Data_Drive/Data/EEGdata/MTB/Model/' 

sio.savemat(mod_vars_loc + 'Binding20TonesModelVars_2.mat',save_dict)





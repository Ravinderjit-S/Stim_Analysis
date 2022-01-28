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



data_loc = '/media/ravinderjit/Data_Drive/Data/EEGdata/MTB/Binding'
pickle_loc = data_loc + '/Pickles/'

fig_loc =  '/media/ravinderjit/Data_Drive/Data/Figures/MTBproj/Binding/'

Subjects = ['S072','S078','S088','S207','S211', 'S259', 'S268', 'S269', 
            'S270', 'S271', 'S272', 'S273', 'S274','S277','S279','S282',
            'S284', 'S285', 'S288', 'S290' ,'S281','S291','S303','S305',
            'S308','S310']

age = [55, 47, 52, 25, 28, 20, 19, 19, 21, 21, 20, 18, 19, 20, 20, 19, 26,
         19,  30, 21, 21, 66, 28, 27, 59, 70]


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

for jj in range(3):
    cnd1 = conds_comp[jj][0]
    cnd2 = conds_comp[jj][1]
    
    onset12_mean = A_evkd_cz[:,:,cnd1].mean(axis=1)
    onset12_sem = A_evkd_cz[:,:,cnd1].std(axis=1) / np.sqrt(A_evkd_cz.shape[1])
    
    ax[jj].plot(t,onset12_mean,label='12')  
    ax[jj].fill_between(t,onset12_mean - onset12_sem, onset12_mean + onset12_sem,alpha=0.5)
    
    onset20_mean = A_evkd_cz[:,:,cnd2].mean(axis=1)
    onset20_sem = A_evkd_cz[:,:,cnd2].std(axis=1) / np.sqrt(A_evkd_cz.shape[1])
    
    ax[jj].plot(t,onset20_mean,label='20')  
    ax[jj].fill_between(t,onset20_mean - onset20_sem, onset20_mean + onset20_sem,alpha=0.5)
    
    ax[jj].ticklabel_format(axis='y',style='sci',scilimits=(0,0))
    ax[jj].set_title(labels[jj])


ax[0].legend()
ax[2].set_xlabel('Time')
#ax[2].set_ylabel('$\mu$V')
fig.suptitle('Average Across Participants')


plt.savefig(os.path.join(fig_loc,'All_12vs20.png'),format='png')

#%% Young Vs Old

cond_bind = ['12 Onset', '20 Onset', '12AB', '12BA', '20AB', '20BA', '12 all','20 all']
        
conds_comp = [[0,1], [2,4], [3,5]]
labels = ['Onset', 'Incoherent to Coherent', 'Coherent to Incoherent']
    
fig,ax = plt.subplots(3,1,sharex=True)

young = np.array(age) < 35
old = np.array(age) > 35

conds_comp = [[1,1], [4,4], [5,5]]
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
fig.suptitle('Average Across Participants')

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

                 
sub_mean = A_evkd_f[:,:,0].mean(axis=1)
sub_sem = A_evkd_f[:,:,0].std(axis=1) / np.sqrt(len(Subjects))
plt.figure()
plt.plot(t_f,sub_mean)
plt.fill_between(t_f, sub_mean-sub_sem,sub_mean+sub_sem,alpha=0.5)

sub_mean = A_evkd_f[:,:,1].mean(axis=1)
sub_sem = A_evkd_f[:,:,1].std(axis=1) / np.sqrt(len(Subjects))

plt.plot(t_f,sub_mean)
plt.fill_between(t_f, sub_mean-sub_sem,sub_mean+sub_sem,alpha=0.5) 

#%% Plot Full time for individuals

fig,ax = plt.subplots(int(np.ceil(len(Subjects)/2)),2,sharex=True, figsize=(14,12))
ax = np.reshape(ax,[int(len(Subjects))])

for sub in range(len(Subjects)):
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

fig,ax = plt.subplots(int(np.ceil(len(Subjects)/2)),2,sharex=True,sharey=False,figsize=(14,12))
ax = np.reshape(ax,[int(len(Subjects))])

cnd_plot = 0
labels = ['Onset', 'AtoB', 'BtoA']
   
fig.suptitle(labels[cnd_plot])

for sub in range(len(Subjects)):
    cnd1 = conds_comp[cnd_plot][0]
    cnd2 = conds_comp[cnd_plot][1]
    
    ax[sub].set_title(Subjects[sub]) 
    
    sub_norm = np.max(np.abs((A_epochs[sub][0].mean(axis=0) + A_epochs[sub][1].mean(axis=0)) / 2))
    sub_norm = 1 # get rid of normalization in plot for now
    
    onset12_mean = (A_epochs[sub][cnd1]).mean(axis=0) / sub_norm
    onset12_sem = (A_epochs[sub][cnd1]).std(axis=0) / np.sqrt(A_epochs[sub][cnd1].shape[0])
    
    ax[sub].plot(t,onset12_mean,label='12')  
    ax[sub].fill_between(t,onset12_mean - onset12_sem, onset12_mean + onset12_sem,alpha=0.5)
    
    onset20_mean = (A_epochs[sub][cnd2]).mean(axis=0) / sub_norm
    onset20_sem = (A_epochs[sub][cnd2]).std(axis=0) / np.sqrt(A_epochs[sub][cnd2].shape[0])
    
    ax[sub].plot(t,onset20_mean, label='20')
    ax[sub].fill_between(t,onset20_mean - onset20_sem, onset20_mean + onset20_sem,alpha=0.5)
    
    #ax[sub].plot(t,onset20_mean-onset12_mean,label='diff',color='k')
    
    #ax[sub].ticklabel_format(axis='y',style='sci',scilimits=(0,0))
    
    
ax[2].legend()
ax[len(Subjects)-1].set_xlabel('Time (sec)')
#ax[len(Subjects)-1].set_ylabel('Norm Amp')
    
plt.savefig(os.path.join(fig_loc, 'Asubj_' + labels[cnd_plot] + '.png'),format='png')

#%% Extract Features from EEG

                     
#Feature List
    #0: onset power (12,20)
    #1: A onset power 12
    #2: A onset power 20
    #3: B onset power 12
    #4: B onset power 20
    #5: onset mean (12,20)
    #6: A12 mean
    #7: A20 mean
    #8: B12 mean
    #9: B20 mean
    
feat_labels = ['Onset', 'A12_On', 'A20_on', 'B12_on', 'B20_on',
               'Onset_mean', 'A12_mean', 'A20_mean', 'B12_mean',
               'B20_mean' ]
  
features = np.zeros([len(Subjects), 10])

onset_inds = [3, 5, 2, 4]

# ta_1 = [0.075, 0.11, 0.16]
# ta_2 = [1.4, 0.16, 0.3]

# tb_1 = [0.1, 0.225, 0.325,0.45]
# tb_2 = [0.225, 0.325, 0.45, 0.9]

t_0 = np.where(t>=0)[0][0]
t_4 = np.where(t>=0.4)[0][0]
t_e = np.where(t>=1)[0][0]

for sub in range(len(Subjects)):
    
    resp = (A_epochs[sub][0].mean(axis=0) + A_epochs[sub][1].mean(axis=0) ) / 2 # mean onset (12,20)
    features[sub,0] = resp[t_0:t_4].var()
    features[sub,5] = np.abs(resp[t_4:t_e].mean())
    
    for on in range(4):
        resp = A_epochs[sub][onset_inds[on]].mean(axis=0)
        features[sub,on+1] = resp[t_0:t_4].var()
        features[sub,on+6] = np.abs(resp[t_4:t_e].mean())
    

    #normalize means??
    # t_0 = 0.5
    # for m in range(5):
    #     t1 = np.where(t_full >= t_0)[0][0]
    #     t2 = np.where(t_full >= t_0 +0.5)[0][0]
        
    #     features[sub,m] = A_evkd_f[t1:t2,sub,0].mean()
    #     features[sub,m+5] = A_evkd_f[t1:t2,sub,1].mean() 
        
    #     t_0 += 1
    
#%% Explore features
    
plt.figure()
plt.scatter(features[:,0],features[:,9])
plt.xlabel(feat_labels[0])
plt.ylabel(feat_labels[9])

plt.figure()
plt.scatter(features[:,1],features[:,6])
plt.xlabel(feat_labels[1])
plt.ylabel(feat_labels[6])

plt.figure()
plt.scatter(features[:,2],features[:,7])
plt.xlabel(feat_labels[2])
plt.ylabel(feat_labels[7])

plt.figure()
plt.scatter(features[:,3],features[:,8])
plt.xlabel(feat_labels[3])
plt.ylabel(feat_labels[8])

plt.figure()
plt.scatter(features[:,4],features[:,9])
plt.xlabel(feat_labels[4])
plt.ylabel(feat_labels[9])

plt.figure()
plt.scatter(features[:,5],features[:,8])
plt.xlabel(feat_labels[5])
plt.ylabel(feat_labels[8])

plt.figure()
plt.scatter(features[:,0],features[:,5])
plt.xlabel(feat_labels[0])
plt.ylabel(feat_labels[5])

#%% Explore normalized features

features_norm = features.copy()
features_norm[:,:5] /= features_norm[:,0][:,np.newaxis]
features_norm[:,5:] /= features_norm[:,5][:,np.newaxis]

f1 =8
f2 =9
plt.figure()
plt.scatter(features_norm[:,f1],features_norm[:,f2])
plt.xlabel(feat_labels[f1])
plt.ylabel(feat_labels[f2])

#%% Compare unnormalized to normalized

f1 = 1
f2 = 2 
fig,ax = plt.subplots(2,1)
ax[0].scatter(features[:,f1],features[:,f2])
ax[1].scatter(features_norm[:,f1],features_norm[:,f2])
ax[1].set_title('Normalized')
ax[1].set_xlabel(feat_labels[f1])
ax[1].set_ylabel(feat_labels[f2])

#%% The 4 B features actually matter after normalization

matter_list = np.array([False,False,False,True,True,False,False,False,True,True])

feat_labels_matter = []
for ml in range(len(matter_list)):
    if matter_list[ml]:
        feat_labels_matter.append(feat_labels[ml])
        
features_matter = features_norm[:,matter_list]

#Add mean grwoth as feature
# g_f2 = features_norm[:,9] / features_norm[:,8] 
# feat_labels_matter.append('B mean growth')
#features_matter = np.concatenate((features_matter,g_f2[:,np.newaxis]),axis=1)


#%% PCA on feature onsets and means

pca = PCA(n_components=3)
pca_on = pca.fit_transform(StandardScaler().fit_transform(features_norm[:,1:5]))
pca_on_expVar = pca.explained_variance_ratio_

plt.figure()
plt.plot(pca.components_.T)
plt.title('On PCA')

pca = PCA(n_components=3)
pca_mn = pca.fit_transform(StandardScaler().fit_transform(features_norm[:,6:]))
pca_mn_expVar = pca.explained_variance_ratio_

plt.figure()
plt.plot(pca.components_.T)
plt.title('Menas pca')

plt.figure()
plt.scatter(pca_mn[:,0], pca_on[:,0])


#%% PCA on matter features

pca = PCA(n_components=2)
pca_feats = pca.fit_transform(StandardScaler().fit_transform(features_matter))
pca_expVar = pca.explained_variance_ratio_

plt.figure()
plt.plot(pca.components_.T)

plt.figure()
plt.plot(pca_feats)




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
spaced_coh = acc_bind[5:7,:].mean(axis=0)

beh_conds = ['2','4','6','8', '4 spaced','6 spaced','8 spaced']

#%% Plot stuff with raw features

for k in range(features.shape[1]):
    
    fig,ax = plt.subplots(3,1,sharex=True)
    ax[0].scatter(features[:,k], thresh_coh)
    ax[0].set_ylabel('Thrsh Coh')
    ax[1].scatter(features[:,k], spaced_coh)
    ax[1].set_ylabel('Spaced Coh')
    ax[2].scatter(features[:,k], consec_coh)
    ax[2].set_ylabel('Consecutive Coh')
    ax[2].set_xlabel(feat_labels[k])

    
#%% Plot Stuff with norm featues

for k in range(features.shape[1]):
    fig,ax = plt.subplots(3,1,sharex=True)
    ax[0].scatter(features_norm[:,k], thresh_coh)
    ax[0].set_ylabel('Thrsh Coh')
    ax[1].scatter(features_norm[:,k], spaced_coh)
    ax[1].set_ylabel('Spaced Coh')
    ax[2].scatter(features_norm[:,k], consec_coh)
    ax[2].set_ylabel('Consecutive Coh')
    ax[2].set_xlabel(feat_labels[k])
    
#%% Is the growth predictive?

g_f1 = features_norm[:,4] / features_norm[:,3]
g_f2 = features_norm[:,9] / features_norm[:,8] 

fig,ax = plt.subplots(3,1,sharex=True)
ax[0].scatter(g_f1, thresh_coh)
ax[0].set_ylabel('Thrsh Coh')
ax[1].scatter(g_f1, spaced_coh)
ax[1].set_ylabel('Spaced Coh')
ax[2].scatter(g_f1, consec_coh)
ax[2].set_ylabel('Consecutive Coh')
ax[2].set_xlabel('B Onset Growth')

fig,ax = plt.subplots(3,1,sharex=True)
ax[0].scatter(g_f2, thresh_coh)
ax[0].set_ylabel('Thrsh Coh')
ax[1].scatter(g_f2, spaced_coh)
ax[1].set_ylabel('Spaced Coh')
ax[2].scatter(g_f2, consec_coh)
ax[2].set_ylabel('Consecutive Coh')
ax[2].set_xlabel('B Mean Growth')



        
#%% PCA features

fig,ax = plt.subplots(3,1)
p_f = 0
ax[0].scatter(pca_feats[:,p_f], thresh_coh)
ax[0].set_ylabel('Thresh Coh')
ax[1].scatter(pca_feats[:,p_f], spaced_coh)
ax[1].set_ylabel('Spaced Coh')
ax[2].scatter(pca_feats[:,p_f], consec_coh)
ax[2].set_ylabel('Consecutive Coh')

ax[2].set_xlabel('PCA ' + str(p_f))


fig,ax = plt.subplots(3,1)
p_f = 0
ax[0].scatter(pca_on[:,p_f], thresh_coh)
ax[0].set_ylabel('Thresh Coh')
ax[1].scatter(pca_on[:,p_f], spaced_coh)
ax[1].set_ylabel('Spaced Coh')
ax[2].scatter(pca_feats[:,p_f], consec_coh)
ax[2].set_ylabel('Consecutive Coh')

ax[2].set_xlabel('PCA_on ' + str(p_f))

fig,ax = plt.subplots(3,1)
p_f = 0
ax[0].scatter(pca_mn[:,p_f], thresh_coh)
ax[0].set_ylabel('Thresh Coh')
ax[1].scatter(pca_mn[:,p_f], spaced_coh)
ax[1].set_ylabel('Spaced Coh')
ax[2].scatter(pca_feats[:,p_f], consec_coh)
ax[2].set_ylabel('Consecutive Coh')

ax[2].set_xlabel('PCA mean ' + str(p_f))




#%% Lapse rate

# plt.figure()
# plt.scatter(acc_bind[2:4,:].mean(axis=0) ,spaced_coh)

plt.figure()
plt.scatter(consec_coh, spaced_coh,label='Spaced')
plt.figure()
plt.scatter(consec_coh, thresh_coh, label='Thresh')


plt.figure()
plt.scatter(thresh_coh,spaced_coh)


#%% Save stuff

save_dict = {'threshCoh': thresh_coh, 'spacedCoh': spaced_coh,
             'consecCoh': consec_coh, 'features': features_matter,
              'feat_labels':feat_labels_matter,
              'pca_feats': pca_feats, 'Subjects': Subjects}

mod_vars_loc = '/media/ravinderjit/Data_Drive/Data/EEGdata/MTB/Model/' 

sio.savemat(mod_vars_loc + 'Binding20TonesModelVars.mat',save_dict)





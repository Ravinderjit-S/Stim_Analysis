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
from scipy import signal



data_loc = '/media/ravinderjit/Data_Drive/Data/EEGdata/MTB/Binding'
pickle_loc = data_loc + '/Pickles/'

fig_loc =  '/media/ravinderjit/Data_Drive/Data/Figures/MTBproj/Binding/'

Subjects = ['S211', 'S259', 'S268', 'S269', 'S270', 'S271', 'S272', 'S273',
            'S274','S277','S279','S282', 'S284', 'S285', 'S288', 'S290']

A_epochs = []

#%% Load data

for subject in Subjects:
    with open(os.path.join(pickle_loc,subject+'_Binding.pickle'),'rb') as file:
        [t, conds_save, epochs_save] = pickle.load(file)
        
    
    A_epochs.append(epochs_save)
  
#%% Get evoked responses
    
A_evkd = np.zeros((len(t),len(Subjects),len(conds_save)))

for sub in range(len(Subjects)):
    for cond in range(len(conds_save)):
        A_evkd[:,sub,cond] = A_epochs[sub][cond].mean(axis=0)
        
#%% Plot Average response across Subjects (unNormalized)

#Conds:
#   0 = 12 Onset
#   1 = 20 Onset
#   3 = 12AB
#   4 = 12BA
#   5 = 20AB
#   6 = 20BA         
        
cond_bind = ['12 Onset', '20 Onset', '12AB', '12BA', '20AB', '20BA']
        
conds_comp = [[0,1], [2,4], [3,5]]
labels = ['Onset', 'Incoherent to Coherent', 'Coherent to Incoherent']
    
fig,ax = plt.subplots(3,1,sharex=True)

for jj in range(3):
    cnd1 = conds_comp[jj][0]
    cnd2 = conds_comp[jj][1]
    
    onset12_mean = A_evkd[:,:,cnd1].mean(axis=1)
    onset12_sem = A_evkd[:,:,cnd1].std(axis=1) / np.sqrt(A_evkd.shape[1])
    
    ax[jj].plot(t,onset12_mean,label='12')  
    ax[jj].fill_between(t,onset12_mean - onset12_sem, onset12_mean + onset12_sem,alpha=0.5)
    
    onset20_mean = A_evkd[:,:,cnd2].mean(axis=1)
    onset20_sem = A_evkd[:,:,cnd2].std(axis=1) / np.sqrt(A_evkd.shape[1])
    
    ax[jj].plot(t,onset20_mean,label='20')  
    ax[jj].fill_between(t,onset20_mean - onset20_sem, onset20_mean + onset20_sem,alpha=0.5)
    
    ax[jj].ticklabel_format(axis='y',style='sci',scilimits=(0,0))
    ax[jj].set_title(labels[jj])


ax[0].legend()
ax[2].set_xlabel('Time')
#ax[2].set_ylabel('$\mu$V')
fig.suptitle('Average Across Participants')


plt.savefig(os.path.join(fig_loc,'All_12vs20.png'),format='png')

#%% Normalize Data to max of of abs of onset response

for sub in range(len(Subjects)):
     sub_norm = np.max(np.abs((A_epochs[sub][0].mean(axis=0) + A_epochs[sub][1].mean(axis=0)) / 2))
     for cnd in range(len(cond_bind)):
        A_epochs[sub][cnd] /= sub_norm
       

#%% Plot all Onsets, AB, or BA

fig,ax = plt.subplots(int(np.ceil(len(Subjects)/2)),2,sharex=True,sharey=True,figsize=(14,12))
ax = np.reshape(ax,[int(len(Subjects))])

cnd_plot = 1
labels = ['Onset', 'AtoB', 'BtoA']
   
fig.suptitle(labels[cnd_plot])

for sub in range(len(Subjects)):
    cnd1 = conds_comp[cnd_plot][0]
    cnd2 = conds_comp[cnd_plot][1]
    
    ax[sub].set_title(Subjects[sub]) 
    
    sub_norm = np.max(np.abs((A_epochs[sub][0].mean(axis=0) + A_epochs[sub][1].mean(axis=0)) / 2))
    
    onset12_mean = (A_epochs[sub][cnd1]).mean(axis=0) / sub_norm
    onset12_sem = (A_epochs[sub][cnd1]).std(axis=0) / np.sqrt(A_epochs[sub][cnd1].shape[0])
    
    ax[sub].plot(t,onset12_mean,label='12')  
    ax[sub].fill_between(t,onset12_mean - onset12_sem, onset12_mean + onset12_sem,alpha=0.5)
    
    onset20_mean = (A_epochs[sub][cnd2]).mean(axis=0) / sub_norm
    onset20_sem = (A_epochs[sub][cnd2]).std(axis=0) / np.sqrt(A_epochs[sub][cnd2].shape[0])
    
    ax[sub].plot(t,onset20_mean, label='20')
    ax[sub].fill_between(t,onset20_mean - onset20_sem, onset20_mean + onset20_sem,alpha=0.5)
    
    ax[sub].plot(t,onset20_mean-onset12_mean,label='diff',color='k')
    
    #ax[sub].ticklabel_format(axis='y',style='sci',scilimits=(0,0))
    
    
ax[2].legend()
ax[len(Subjects)-1].set_xlabel('Time (sec)')
ax[len(Subjects)-1].set_ylabel('Norm Amp')
    
plt.savefig(os.path.join(fig_loc, 'Asubj_' + labels[cnd_plot] + '.png'),format='png')

#%% Extract Features from EEG

n_feats = len(cond_bind) *2 
features = np.zeros([len(Subjects), n_feats]) # Subjects x conditions x features for each condition
                     
#Feature List
 #0-5: cond_bind means 
 #6: cond_bind max of abs



for sub in range(len(Subjects)):
    
    t1 = np.where(t>=0.4)[0][0]
    t2 = np.where(t>=1.0)[0][0]
    for feat in range(len(cond_bind)): # First 6 features are mean
        mean = (A_epochs[sub][feat].mean(axis=0)[t1:t2]).mean() 
        features[sub,feat] = mean
        
        
    t1 = np.where(t>=.150)[0][0]
    t2 = np.where(t>=0.3)[0][0]
    for feat in range(len(cond_bind),2*len(cond_bind)): # Next 6 are max
        mx = np.max(np.abs(A_epochs[sub][feat - 6].mean(axis=0)[t1:t2]))
        features[sub,feat] = mx
    
    
B_present = np.array([1,1,0,1,1,0,0,0,0,0,1,0,0,0,0,1],dtype=bool)

#%% Use PCA to condense temporal Features from full time waveform

t1 = np.where(t>=0.4)[0][0]
t2 = np.where(t>=1)[0][0]
tp = t[t1:t2]

Evkd_all = np.zeros((tp.size*6,len(Subjects)))
Evkd_conds = np.zeros((tp.size,len(Subjects),len(conds_save)))


for sub in range(len(Subjects)):
    A_evkd = np.array([])
    for cond in range(len(conds_save)):
        evkd = A_epochs[sub][cond].mean(axis=0)[t1:t2]
        #evkd = signal.decimate(evkd,16) --- computer can handle but okay to do this if want to
        A_evkd = np.concatenate((A_evkd,evkd))
        Evkd_conds[:,sub,cond] = evkd
    
    Evkd_all[:,sub] = A_evkd
    

X_pca = []
X_expVar = []
X_comp = []

pca = PCA(n_components=10)
X = pca.fit_transform(Evkd_all.T)

X_pca.append(X)
X_expVar.append(pca.explained_variance_ratio_)
X_comp.append(pca.components_)

X = pca.fit_transform(Evkd_all[:len(tp)*2,:].T)
X_pca.append(X)
X_expVar.append(pca.explained_variance_ratio_)
X_comp.append(pca.components_)

X = pca.fit_transform(Evkd_all[len(tp)*2:,:].T)
X_pca.append(X)
X_expVar.append(pca.explained_variance_ratio_)
X_comp.append(pca.components_)


for cond in range(len(conds_save)):
    X = pca.fit_transform(Evkd_conds[:,:,cond].T)
    X_pca.append(X)
    X_expVar.append(pca.explained_variance_ratio_)
    X_comp.append(pca.components_)


#X
#0: All 6 conds
#1: Onset conds (first 2 conds)
#2: Other 4 conds (4 binding conditions)
#3-8 : Individual conds
#9: 20AB - 12 AB
#10: 20BA - 12BA

tend = tp[-1]

big_t = np.concatenate([tp, tp+ tend, tp + tend*2, tp + tend*3, tp + tend*4, tp + tend*5])
plt.figure()
plt.plot(big_t,X_comp[0][0:3,:].T)

plt.figure()
plt.plot(big_t[:len(tp)*2],X_comp[1][0:2,:].T)
plt.plot(big_t[:len(tp)*2],np.sum(X_comp[1][0:4],axis=0),color='k')

plt.figure()
plt.plot(big_t[len(tp)*2:],X_comp[2][0:2,:].T)
plt.plot(big_t[len(tp)*2:],np.sum(X_comp[2][0:4],axis=0),color='k')

fix,ax = plt.subplots(3,2)
ax = np.reshape(ax,len(conds_save))
for cond in range(len(conds_save)):
    ax[cond].plot(tp,X_comp[cond+3][0:2,:].T)
    ax[cond].plot(tp,np.mean(X_comp[cond+3][0:2,:],axis=0),color='k')
    
    

plt.figure()
plt.scatter(X_pca[1][:,0], X_pca[2][:,0])


fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(X_pca[1][:,0], X_pca[1][:,1], X_pca[1][:,2])

plt.figure()
plt.scatter(np.mean(X_pca[1][:,0:3],axis=1), np.mean(X_pca[2][:,0:3],axis=1))


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

consec_coh = acc_bind[1:2,:].mean(axis=0)
spaced_coh = acc_bind[5:6,:].mean(axis=0)

beh_conds = ['2','4','6','8', '4 spaced','6 spaced','8 spaced']

#%% Plot stuff

# for beh_ind in range(len(beh_conds)):
#     fig,ax = plt.subplots(3,2)
#     ax = np.reshape(ax,6)
#     for cnd in range(len(cond_bind)):
#         ax[cnd].scatter(features[:,cnd],acc_bind[beh_ind,:])
#         ax[cnd].set_title(cond_bind[cnd])
    
#     ax[4].set_xlabel('EEG Feat')
#     ax[4].set_ylabel('Beh Accuracy')
#     fig.suptitle(beh_conds[beh_ind])


fig,ax = plt.subplots(2,2)
ax = np.reshape(ax,4)
for cnd in range(2,len(cond_bind)):
    ax[cnd-2].scatter(features[:,cnd + len(cond_bind)],consec_coh)
    ax[cnd-2].set_title(cond_bind[cnd])
    
ax[3].set_xlabel('EEG Feat')
ax[3].set_ylabel('Beh Accuracy')
fig.suptitle('Consecutive Coherence')


fig,ax = plt.subplots(2,2)
ax = np.reshape(ax,4)
for cnd in range(2,len(cond_bind)):
    ax[cnd-2].scatter(features[:,cnd + len(cond_bind)],spaced_coh)
    ax[cnd-2].set_title(cond_bind[cnd])
    
ax[3].set_xlabel('EEG Feat')
ax[3].set_ylabel('Beh Accuracy')
fig.suptitle('Spaced Coherence')
  
plt.figure()
plt.scatter(spaced_coh,consec_coh)
plt.xlabel('Spaced Coh')
plt.ylabel('Consec Coh')


plt.figure()
plt.scatter(features[B_present,10],consec_coh[B_present])
plt.scatter(features[~B_present,10],consec_coh[~B_present])

plt.figure()
plt.scatter(features[B_present,10],spaced_coh[B_present])
plt.scatter(features[~B_present,10],spaced_coh[~B_present])


plt.figure()
plt.scatter(features[:,11] / features[:,9], consec_coh)

plt.figure()
plt.scatter(features[:,11] / features[:,9], spaced_coh)

plt.figure()
plt.scatter(features[:,11] - features[:,9], spaced_coh)


fig,ax = plt.subplots(2,1)
ax[0].scatter(X[:,0:2].mean(axis=1),consec_coh)
ax[0].set_ylabel('Consecutive Accuracy')
ax[1].scatter(X[:,0:2].mean(axis=1),spaced_coh)
ax[1].set_ylabel('Spaced Accuracy')


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 20:54:08 2021

@author: ravinderjit
"""

import os
import pickle
import numpy as np
import scipy as sp
import mne
import matplotlib.pyplot as plt
from scipy.signal import freqz
from scipy.io import loadmat
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA


data_loc = os.path.abspath('/media/ravinderjit/Data_Drive/Data/EEGdata/DynamicBinaural/Pickles_32/SystemFuncs32')
fig_path = os.path.abspath('/media/ravinderjit/Data_Drive/Data/Figures/DynBin')


Subjects = ['S001','S132','S203','S204','S205','S206','S207','S208','S211']
# Subjects = ['S132','S203','S204','S205','S206','S207','S208','S211']

A_pca_space_IAC = []
A_pca_space_ITD = []
A_ica_space_IAC = []
A_ica_space_ITD = []

A_pca_f_IAC = []
A_pca_f_ITD = []
A_pca_f_IAC_nf = []
A_pca_f_ITD_nf = []

A_pca_space_IAC_nf = []
A_pca_space_ITD_nf = []
A_ica_space_IAC_nf = []
A_ica_space_ITD_nf = []

A_pca_coeff_IAC = []


A_pca_coeff_ITD = []
A_ica_coeff_IAC = []
A_ica_coeff_ITD = []

A_pca_coeff_IAC_nf = []
A_pca_coeff_ITD_nf = []
A_ica_coeff_IAC_nf = []
A_ica_coeff_ITD_nf = []

A_IAC_Ht = []
A_ITD_Ht = []


A_pca_expVar_IAC = np.zeros([len(Subjects),2])
A_pca_expVar_ITD = np.zeros([len(Subjects),2])

with open('/media/ravinderjit/Data_Drive/Data/EEGdata/DynamicBinaural/Pickles_32/S211_DynBin.pickle','rb') as f:
    IAC_epochs, ITD_epochs = pickle.load(f)
    
    
fs = int(IAC_epochs.info['sfreq'])


#%% Load Data
for sub in range(len(Subjects)):
    Subject = Subjects[sub]
    with open(os.path.join(data_loc, Subject+'_DynBin_SysFunc.pickle'),'rb') as file:     
              [t,f,IAC_Ht,ITD_Ht,IAC_Htnf,ITD_Htnf,IAC_Hf,ITD_Hf,
                      pca_space_IAC,pca_f_IAC,pca_coeff_IAC,pca_expVar_IAC,
                      pca_space_ITD,pca_f_ITD,pca_coeff_ITD,pca_expVar_ITD, 
                      pca_space_IAC_nf,pca_f_IAC_nf,pca_coeffs_IAC_nf,
                      pca_expVar_IAC_nf, pca_space_ITD_nf,pca_f_ITD_nf,
                      pca_coeffs_ITD_nf,pca_expVar_ITD_nf, ica_space_ITD,
                      ica_f_ITD,ica_coeff_ITD, ica_space_IAC, ica_f_IAC, 
                      ica_coeff_IAC, ica_space_ITD_nf,ica_f_ITD_nf,ica_coeffs_ITD_nf,
                      ica_space_IAC_nf, ica_f_IAC_nf,ica_coeffs_IAC_nf] = pickle.load(file)
             
    # with open(os.path.join(data_loc, Subject+'_DynBin_SysFunc.pickle'),'rb') as file:
    #     [t,IAC_Ht,ITD_Ht,IAC_Htnf,ITD_Htnf,
    #         pca_space_IAC,pca_coeff_IAC,pca_expVar_IAC,
    #         pca_space_ITD,pca_coeff_ITD,pca_expVar_ITD, 
    #         pca_space_IAC_nf,pca_coeffs_IAC_nf,
    #         pca_expVar_IAC_nf, pca_space_ITD_nf,
    #         pca_coeffs_ITD_nf,pca_expVar_ITD_nf, ica_space_ITD,
    #         ica_coeff_ITD, ica_space_IAC, ica_coeff_IAC, 
    #         ica_space_ITD_nf,ica_f_ITD_nf,ica_coeffs_ITD_nf,
    #         ica_space_IAC_nf,ica_coeffs_IAC_nf] = pickle.load(file)
             
    A_pca_space_IAC.append(pca_space_IAC)
    A_pca_space_ITD.append(pca_space_ITD)
    A_ica_space_IAC.append(ica_space_IAC)
    A_ica_space_ITD.append(ica_space_ITD)
    
    A_pca_space_IAC_nf.append(pca_space_IAC_nf)
    A_pca_space_ITD_nf.append(pca_space_ITD_nf)
    A_ica_space_IAC_nf.append(ica_space_IAC_nf)
    A_ica_space_ITD_nf.append(ica_space_ITD_nf)
    
    # A_pca_f_IAC.append(pca_f_IAC)
    # A_pca_f_ITD.append(pca_f_ITD)    
    # A_pca_f_IAC_nf.append(pca_f_IAC_nf)
    # A_pca_f_ITD_nf.append(pca_f_ITD_nf)
    
    A_pca_coeff_IAC.append(pca_coeff_IAC)
    A_pca_coeff_ITD.append(pca_coeff_ITD)
    A_ica_coeff_IAC.append(ica_coeff_IAC)
    A_ica_coeff_ITD.append(ica_coeff_ITD)
    
    A_pca_coeff_IAC_nf.append(pca_coeffs_IAC_nf)
    A_pca_coeff_ITD_nf.append(pca_coeffs_ITD_nf)
    A_ica_coeff_IAC_nf.append(ica_coeffs_IAC_nf)
    A_ica_coeff_ITD_nf.append(ica_coeffs_ITD_nf)
    
    A_pca_expVar_IAC[sub,:] = pca_expVar_IAC
    A_pca_expVar_ITD[sub,:] = pca_expVar_ITD
    
    A_IAC_Ht.append(IAC_Ht)
    A_ITD_Ht.append(ITD_Ht)
             
 
#%% Check sign of decomposition
# PCA just depends on covariance so multiplying by -1 is incosequential. But it is helpful for topomap visualization and averaging
Aud_chs = [4,8,12,21,25,30,31]

for sub in range(len(Subjects)):
    if A_pca_coeff_IAC[sub][0,Aud_chs].mean() < (A_pca_coeff_IAC[sub].mean(axis=1)[0]):
        A_pca_coeff_IAC[sub] = - A_pca_coeff_IAC[sub]
        A_pca_space_IAC[sub] = - A_pca_space_IAC[sub]
    if A_pca_coeff_ITD[sub][0,Aud_chs].mean() < (A_pca_coeff_ITD[sub].mean(axis=1)[0]):
        A_pca_coeff_ITD[sub] = - A_pca_coeff_ITD[sub]
        A_pca_space_ITD[sub] = - A_pca_space_ITD[sub]
    for nf in range(len(A_pca_coeff_IAC_nf[sub])):
        if A_pca_coeff_IAC_nf[sub][nf][0,30:].mean() < (A_pca_coeff_IAC_nf[sub][nf].mean(axis=1)[0] - A_pca_coeff_IAC_nf[sub][nf].std(axis=1)[0]):
            A_pca_coeff_IAC_nf[sub][nf] = - A_pca_coeff_IAC_nf[sub][nf]
            A_pca_space_IAC_nf[sub][nf] = - A_pca_space_IAC_nf[sub][nf]
    for nf in range(len(A_pca_coeff_ITD_nf[sub])):
        if A_pca_coeff_ITD_nf[sub][nf][0,30:].mean() < (A_pca_coeff_ITD_nf[sub][nf].mean(axis=1)[0] - A_pca_coeff_ITD_nf[sub][nf].std(axis=1)[0]):
            A_pca_coeff_ITD_nf[sub][nf] = - A_pca_coeff_ITD_nf[sub][nf]
            A_pca_space_ITD_nf[sub][nf] = - A_pca_space_ITD_nf[sub][nf]
            
        
        
#Changing S001 by visual inspection
A_pca_space_IAC[0] = - A_pca_space_IAC[0]
A_pca_coeff_IAC[0] = - A_pca_coeff_IAC[0]


#%% Plot Topomaps
ch_picks = range(32)            
sbp = [3 , 3]
vmin = A_pca_coeff_IAC[0].mean(axis=1)[0] - 2*A_pca_coeff_IAC[0].std(axis=1)[0]
vmax = A_pca_coeff_IAC[0].mean(axis=1)[0] + 2*A_pca_coeff_IAC[0].std(axis=1)[0]
plt.figure()
pca_coeffs_IAC_c1 = np.zeros([len(ch_picks),len(Subjects)])
for s in range(sbp[0]*sbp[1]):
   plt.subplot(sbp[0],sbp[1],s+1)
   mne.viz.plot_topomap(A_pca_coeff_IAC[s][0,:], mne.pick_info(IAC_epochs.info, ch_picks),vmin=vmin,vmax=vmax)
   plt.title(Subjects[s])
   pca_coeffs_IAC_c1[:,s] = A_pca_coeff_IAC[s][0,:]
   
plt.figure()
mne.viz.plot_topomap(pca_coeffs_IAC_c1.mean(axis=1), mne.pick_info(IAC_epochs.info, ch_picks),vmin=vmin,vmax=vmax)
plt.title('IAC from averaging pca coefficients')

# Get all IAC nf coeffs
num_nfs = len(A_pca_coeff_IAC_nf[0])
IAC_coeffs_nf = np.zeros([32,num_nfs,len(Subjects)])
for sub in range(len(Subjects)):
    for nf in range(num_nfs):
        IAC_coeffs_nf[:,nf,sub] = A_pca_coeff_IAC_nf[sub][nf][0,:]
        
plt.figure()
for s in range(sbp[0]*sbp[1]):
   plt.subplot(sbp[0],sbp[1],s+1)
   mne.viz.plot_topomap(IAC_coeffs_nf[:,:,s].mean(axis=1), mne.pick_info(IAC_epochs.info, ch_picks),vmin=vmin,vmax=vmax)
   plt.title(Subjects[s] + ' nf')
   
plt.figure()
mne.viz.plot_topomap(IAC_coeffs_nf.mean(axis=2).mean(axis=1), mne.pick_info(IAC_epochs.info, ch_picks),vmin=vmin,vmax=vmax)
plt.title('IAC Nf')


vmin = A_pca_coeff_ITD[0].mean(axis=1)[0] - 2*A_pca_coeff_ITD[0].std(axis=1)[0]
vmax = A_pca_coeff_ITD[0].mean(axis=1)[0] + 2*A_pca_coeff_ITD[0].std(axis=1)[0]
pca_coeffs_ITD_c1 = np.zeros([len(ch_picks),len(Subjects)])
plt.figure()
for s in range(sbp[0]*sbp[1]):
   plt.subplot(sbp[0],sbp[1],s+1)
   mne.viz.plot_topomap(A_pca_coeff_ITD[s][0,:], mne.pick_info(ITD_epochs.info, ch_picks),vmin=vmin,vmax=vmax)
   plt.title(Subjects[s])
   pca_coeffs_ITD_c1[:,s] = A_pca_coeff_ITD[s][0,:]
   
plt.figure()
mne.viz.plot_topomap(pca_coeffs_ITD_c1.mean(axis=1), mne.pick_info(IAC_epochs.info, ch_picks),vmin=vmin,vmax=vmax)
plt.title('ITD')

# Get all ITD nf coeffs
num_nfs = len(A_pca_coeff_ITD_nf[0])
ITD_coeffs_nf = np.zeros([32,num_nfs,len(Subjects)])
for sub in range(len(Subjects)):
    for nf in range(num_nfs):
        ITD_coeffs_nf[:,nf,sub] = A_pca_coeff_ITD_nf[sub][nf][0,:]

plt.figure()
for s in range(sbp[0]*sbp[1]):
   plt.subplot(sbp[0],sbp[1],s+1)
   mne.viz.plot_topomap(ITD_coeffs_nf[:,:,s].mean(axis=1), mne.pick_info(ITD_epochs.info, ch_picks),vmin=vmin,vmax=vmax)
   plt.title(Subjects[s] + ' nf')
   
plt.figure()
mne.viz.plot_topomap(ITD_coeffs_nf.mean(axis=2).mean(axis=1), mne.pick_info(ITD_epochs.info, ch_picks),vmin=vmin,vmax=vmax)
plt.title('ITD Nf')



             
ch_picks = range(32)            
sbp = [3 , 3]
vmin = A_ica_coeff_IAC[0].mean() - 2*A_ica_coeff_IAC[0].std()
vmax = A_ica_coeff_IAC[0].mean() + 2*A_ica_coeff_IAC[0].std()
plt.figure()
for s in range(sbp[0]*sbp[1]):
   plt.subplot(sbp[0],sbp[1],s+1)    
   mne.viz.plot_topomap(A_ica_coeff_IAC[s][0,:], mne.pick_info(IAC_epochs.info, ch_picks),vmin=vmin,vmax=vmax)
    

ch_picks = range(32)            
sbp = [3 , 3]
vmin = A_ica_coeff_ITD[0].mean() - 2*A_ica_coeff_ITD[0].std()
vmax = A_ica_coeff_ITD[0].mean() + 2*A_ica_coeff_ITD[0].std()
plt.figure()
for s in range(sbp[0]*sbp[1]):
   plt.subplot(sbp[0],sbp[1],s+1)
   mne.viz.plot_topomap(A_ica_coeff_ITD[s][0,:], mne.pick_info(ITD_epochs.info, ch_picks),vmin=vmin,vmax=vmax)
   

#%% Get response from average PCA weights
Avg_weights_IAC = pca_coeffs_IAC_c1.mean(axis=1)

Avg_pca_sp = np.zeros([t.size,len(Subjects)])
for s in range(len(Subjects)):
    Avg_pca_sp[:,s] = np.matmul(Avg_weights_IAC, A_IAC_Ht[s]-A_IAC_Ht[s].mean(axis=1)[:,np.newaxis])


plt.figure()
plt.plot(t,Avg_pca_sp)
# plt.plot(t,A_pca_space_IAC[0][:,0])

#%% Get response from pca on average response

#get Ht into numpy vector
Anp_Ht_IAC = np.zeros([A_IAC_Ht[0].shape[0],A_IAC_Ht[0].shape[1],len(Subjects)])
Anp_Ht_ITD = np.zeros([A_ITD_Ht[0].shape[0],A_ITD_Ht[0].shape[1],len(Subjects)])
for s in range(len(Subjects)):
    Anp_Ht_IAC[:,:,s] = A_IAC_Ht[s]
    Anp_Ht_ITD[:,:,s] = A_ITD_Ht[s]
    
Ht_avg_IAC = Anp_Ht_IAC.mean(axis=2)
Ht_avg_ITD = Anp_Ht_ITD.mean(axis=2)

pca = PCA(n_components=1)
pca_sp_Htavg_IAC = pca.fit_transform(Ht_avg_IAC.T)
pca_Htavg_IACcoeffs = pca.components_
pca_Htavg_IACexpVar = pca.explained_variance_ratio_

pca = PCA(n_components=1)
pca_sp_Htavg_ITD = pca.fit_transform(Ht_avg_ITD.T)
pca_Htavg_ITDcoeffs = pca.components_
pca_Htavg_ITDexpVar = pca.explained_variance_ratio_

vmin = A_pca_coeff_IAC[0].mean(axis=1)[0] - 2*A_pca_coeff_IAC[0].std(axis=1)[0]
vmax = A_pca_coeff_IAC[0].mean(axis=1)[0] + 2*A_pca_coeff_IAC[0].std(axis=1)[0]
plt.figure()
mne.viz.plot_topomap(pca_Htavg_IACcoeffs.squeeze(), mne.pick_info(IAC_epochs.info, ch_picks),vmin=vmin,vmax=vmax)
plt.title('IAC: Avg Subjects then calculate coeffs')

vmin = A_pca_coeff_ITD[0].mean(axis=1)[0] - 2*A_pca_coeff_ITD[0].std(axis=1)[0]
vmax = A_pca_coeff_ITD[0].mean(axis=1)[0] + 2*A_pca_coeff_ITD[0].std(axis=1)[0]
plt.figure()
mne.viz.plot_topomap(pca_Htavg_ITDcoeffs.squeeze(), mne.pick_info(ITD_epochs.info, ch_picks),vmin=vmin,vmax=vmax)
plt.title('ITD: Avg Subjects then calculate coeffs')


# ica = FastICA(n_components=1)
# ica_sp_Htavg = ica.fit_transform(Ht_avg.T)
# ica_Htavg_coeffs = ica.whitening_

# vmin = A_ica_coeff_IAC[0].mean(axis=1)[0] - 2*A_ica_coeff_IAC[0].std(axis=1)[0]
# vmax = A_ica_coeff_IAC[0].mean(axis=1)[0] + 2*A_ica_coeff_IAC[0].std(axis=1)[0]
# plt.figure()
# mne.viz.plot_topomap(ica_Htavg_coeffs.squeeze(), mne.pick_info(IAC_epochs.info, ch_picks),vmin=vmin,vmax=vmax)
# plt.title('IAC: ica Avg Subjects then calculate coeffs')


plt.figure()
plt.plot(t,pca_sp_Htavg_IAC, label='PCA on average')
#plt.plot(t,ica_sp_Htavg)
plt.plot(t,pca_IAC_comp1.mean(axis=1),color='k',label='Average 1st pca components')
plt.legend()

plt.figure()
plt.plot(t,pca_sp_Htavg_ITD, label='PCA on average')
#plt.plot(t,ica_sp_Htavg)
plt.plot(t,pca_ITD_comp1.mean(axis=1),color='k',label='Average 1st pca components')
plt.legend()


#%% Plot Time Domain
   
# Get DData into numpy vectors
pca_IAC_comp1 = np.zeros([t.size,len(Subjects)])
pca_ITD_comp1 = np.zeros([t.size,len(Subjects)])
for s in range(len(Subjects)):
    pca_IAC_comp1[:,s] = A_pca_space_IAC[s][:,0]
    pca_ITD_comp1[:,s] = A_pca_space_ITD[s][:,0]
    
    
plt.figure()
plt.title('IAC')
plt.plot(t,pca_IAC_comp1.mean(axis=1),color='k',linewidth=2,label='mean')
for s in range(len(Subjects)):
    plt.plot(t,pca_IAC_comp1[:,s],label=Subjects[s])
plt.legend()    

plt.figure()
plt.title('ITD')
for s in range(len(Subjects)):
    plt.plot(t,pca_ITD_comp1[:,s],label=Subjects[s])
plt.plot(t,pca_ITD_comp1.mean(axis=1),color='k',linewidth=2,label='mean')
plt.legend()


plt.figure()
plt.title('IAC')
A_NF_mean = np.zeros([t.size,len(Subjects)])
for s in range(sbp[0]*sbp[1]):
   plt.subplot(sbp[0],sbp[1],s+1)
   plt.plot(t,pca_IAC_comp1[:,s])
   NF_s = np.zeros([t.size,len(A_pca_space_IAC_nf[s])])
   for n in range(len(A_pca_space_IAC_nf[s])):
       NF_s[:,n]= A_pca_space_IAC_nf[s][n][:,0]

   NF_std = NF_s.std(axis=1) #/ np.sqrt(NF_s.shape[1])
   NF_mean = NF_s.mean(axis=1)
   A_NF_mean[:,s] = NF_mean
   plt.plot(t,NF_mean,color='grey',linewidth=2)
   plt.fill_between(t,NF_mean+1.96*NF_std,NF_mean - 1.96 * NF_std,color='grey',alpha=0.3)
   plt.title(Subjects[s])

   
num_nfs = len(A_pca_space_IAC_nf[0])
All_nfs = np.zeros([t.size,num_nfs*len(Subjects)])
for s in range(len(Subjects)):
    for nf in range(num_nfs):
        All_nfs[:,s*num_nfs+nf] = A_pca_space_IAC_nf[s][nf][:,0]
        
        
        
plt.figure()
resp_mean = pca_IAC_comp1.mean(axis=1)
resp_se =  pca_IAC_comp1.std(axis=1) / np.sqrt(pca_IAC_comp1.shape[1])
plt.plot(t,resp_mean,linewidth=2)
plt.fill_between(t,resp_mean+1.96*resp_se,resp_mean-1.96*resp_se)
# NF_mean = A_NF_mean.mean(axis=1) 
# NF_std = A_NF_mean.std(axis=1) 
NF_mean = All_nfs.mean(axis=1) 
NF_std = All_nfs.std(axis=1) / np.sqrt(pca_IAC_comp1.shape[1])
plt.plot(t,NF_mean,color='grey',linewidth=2)
plt.fill_between(t,NF_mean+1.96*NF_std,NF_mean-1.96*NF_std,color='grey',alpha=0.3)


vmin = A_pca_coeff_ITD[0].mean(axis=1)[0] - 2*A_pca_coeff_ITD[0].std(axis=1)[0]
vmax = A_pca_coeff_ITD[0].mean(axis=1)[0] + 2*A_pca_coeff_ITD[0].std(axis=1)[0]


plt.figure()
plt.title('ITD')
for s in range(sbp[0]*sbp[1]):
   plt.subplot(sbp[0],sbp[1],s+1)
   plt.plot(t,pca_ITD_comp1[:,s])
   NF_s = np.zeros([t.size,len(A_pca_space_ITD_nf[s])])
   for n in range(len(A_pca_space_ITD_nf[s])):
       NF_s[:,n] = A_pca_space_ITD_nf[s][n][:,0]
   NF_std = NF_s.std(axis=1)
   NF_mean = NF_s.mean(axis=1)
   plt.plot(t,NF_mean,color='grey',linewidth=2)
   plt.fill_between(t,NF_mean+1.96*NF_std,NF_mean-1.96*NF_std,color='grey',alpha=0.3)
   plt.title(Subjects[s])
         
       
   
num_nfs = len(A_pca_space_ITD_nf[0])
All_nfs_ITD = np.zeros([t.size,num_nfs*len(Subjects)])
for s in range(len(Subjects)):
    for nf in range(num_nfs):
        All_nfs_ITD[:,s*num_nfs+nf] = A_pca_space_ITD_nf[s][nf][:,0]
        
        
plt.figure()
resp_mean = pca_ITD_comp1.mean(axis=1)
resp_se =  pca_ITD_comp1.std(axis=1) / np.sqrt(pca_ITD_comp1.shape[1])
plt.plot(t,resp_mean,linewidth=2)
plt.fill_between(t,resp_mean+1.96*resp_se,resp_mean-1.96*resp_se)
# NF_mean = A_NF_mean.mean(axis=1) 
# NF_std = A_NF_mean.std(axis=1) 
NF_mean = All_nfs_ITD.mean(axis=1) 
NF_std = All_nfs_ITD.std(axis=1) / np.sqrt(pca_IAC_comp1.shape[1])
plt.plot(t,NF_mean,color='grey',linewidth=2)
plt.fill_between(t,NF_mean+1.96*NF_std,NF_mean-1.96*NF_std,color='grey',alpha=0.3)



    

#%% Frequency domain && Compare to behavioral IAC
       
beh_dataPath = '/media/ravinderjit/Data_Drive/Data/BehaviorData/IACbehavior/'
beh_IACsq = loadmat(os.path.abspath(beh_dataPath+'IACsquareTone_Processed.mat'))

f_beh = beh_IACsq['WindowSizes'][0][1:]**-1
SNRs_beh = beh_IACsq['AcrossSubjectsSNR'][1:] - beh_IACsq['AcrossSubjectsSNR'][0]
AcrossSubjectSEM = beh_IACsq['AcrossSubjectsSEM'][1:]

mask6dB_beh = (SNRs_beh > SNRs_beh.max()-6).squeeze()
       
b = pca_IAC_comp1.mean(axis=1)
#b = b[:int(np.round(.2*fs))]

w,h = freqz(b,a=1,worN=4000,fs=fs)

phase_IAC = np.unwrap(np.angle(h))

mask6dB_physio = (20*np.log10(np.abs(h))) > (20*np.log10(np.abs(h))).max()-6

fig,ax = plt.subplots()
ax.plot(w[mask6dB_physio],20*np.log10(np.abs(h[mask6dB_physio])))
ax.set_xscale('log')
ax.set_xlim([0.5,20])
ax2 = ax.twinx()
ax2.errorbar(f_beh[mask6dB_beh],SNRs_beh[mask6dB_beh],AcrossSubjectSEM[mask6dB_beh],color='r')
ax2.set_xscale('log')

plt.figure()
plt.plot(w,np.abs(h))
plt.xlim([0,20])

f_2_ind = np.where(w>=2)[0][0]
f_8_ind = np.where(w>=8)[0][0]

coeff = np.polyfit(w[f_2_ind:f_8_ind],phase_IAC[f_2_ind:f_8_ind],deg=1)
GD_line = coeff[0] * w[f_2_ind:f_8_ind] + coeff[1]
GD = -coeff[0] / (2*np.pi)

plt.figure()
plt.plot(w,phase_IAC)
plt.plot(w[f_2_ind:f_8_ind],GD_line,color='k')
plt.xlim([0,20])
plt.ylim([-30,5])



b_ITD = pca_ITD_comp1.mean(axis=1)
#b_ITD = b_ITD[:int(np.round(.2*fs))]

w_itd,h_itd = freqz(b_ITD,a=1,worN=4000,fs=fs)
phase_itd = np.unwrap(np.angle(h_itd))

plt.figure()
plt.plot(w_itd,np.abs(h_itd))
plt.xlim([0,20])

plt.figure()
plt.plot(w_itd,np.abs(h_itd))
plt.plot(w,np.abs(h))
plt.xlim([0,20])

plt.figure()
plt.plot(w,20*np.log10(np.abs(h)),label='IAC')
plt.plot(w_itd, 20*np.log10(np.abs(h_itd)),label='ITD')
plt.xlim([0,20])
plt.ylim([-40,5])
plt.legend()

f_2_ind = np.where(w_itd>=2)[0][0]
f_8_ind = np.where(w_itd>=8)[0][0]

coeff = np.polyfit(w_itd[f_2_ind:f_8_ind],phase_itd[f_2_ind:f_8_ind],deg=1)
GD_line = coeff[0] * w_itd[f_2_ind:f_8_ind] + coeff[1]
GD = -coeff[0] / (2*np.pi)

plt.figure()
plt.plot(w_itd,phase_itd)
plt.plot(w_itd[f_2_ind:f_8_ind],GD_line,color='black')
plt.xlim([0,20])
plt.ylim([-5,2])







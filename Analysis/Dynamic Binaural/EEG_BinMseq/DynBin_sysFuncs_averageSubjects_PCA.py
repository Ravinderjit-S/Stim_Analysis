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


#data_loc = os.path.abspath('/media/ravinderjit/Data_Drive/Data/EEGdata/DynamicBinaural/Pickles_32/SystemFuncs32_2')
data_loc = os.path.abspath('/media/ravinderjit/Data_Drive/Data/EEGdata/DynamicBinaural/Pickles_32/SystemFuncs32_IIR/SysFunc_32_IIR')
fig_path = os.path.abspath('/media/ravinderjit/Data_Drive/Data/Figures/DynBin')


Subjects = ['S001','S132','S203','S204','S205','S206','S207','S208','S211']
Subjects = ['S001','S132','S203','S204','S205','S206','S207','S208','S211']


A_IAC_Ht = []
A_ITD_Ht = []

A_IAC_Ht_nf = []
A_ITD_Ht_nf = []

ch_picks = np.arange(32)


A_pca_expVar_IAC = np.zeros([len(Subjects),2])
A_pca_expVar_ITD = np.zeros([len(Subjects),2])

with open('/media/ravinderjit/Data_Drive/Data/EEGdata/DynamicBinaural/Pickles_32/S211_DynBin.pickle','rb') as f:
    IAC_epochs, ITD_epochs = pickle.load(f)
    
    
fs = int(IAC_epochs.info['sfreq'])
info_obj = mne.pick_info(ITD_epochs.info, ch_picks)
del IAC_epochs, ITD_epochs

#%% Load Data
for sub in range(len(Subjects)):
    Subject = Subjects[sub]
    with open(os.path.join(data_loc, Subject+'_DynBin_SysFunc.pickle'),'rb') as file:     
        [t, IAC_Ht, ITD_Ht, IAC_Htnf, ITD_Htnf] = pickle.load(file)
        
        print(sub)
        
        # [t,IAC_Ht,ITD_Ht,IAC_Htnf,ITD_Htnf,
        #     pca_space_IAC,pca_coeff_IAC,pca_expVar_IAC,
        #     pca_space_ITD,pca_coeff_ITD,pca_expVar_ITD, 
        #     pca_space_IAC_nf,pca_coeffs_IAC_nf,
        #     pca_expVar_IAC_nf, pca_space_ITD_nf,
        #     pca_coeffs_ITD_nf,pca_expVar_ITD_nf, ica_space_ITD,
        #     ica_coeff_ITD, ica_space_IAC, ica_coeff_IAC, 
        #     ica_space_ITD_nf,ica_f_ITD_nf,ica_coeffs_ITD_nf,
        #     ica_space_IAC_nf,ica_coeffs_IAC_nf]= pickle.load(file)
        
              # [t,f,IAC_Ht,ITD_Ht,IAC_Htnf,ITD_Htnf,IAC_Hf,ITD_Hf,
              #   pca_space_IAC,pca_f_IAC,pca_coeff_IAC,pca_expVar_IAC,
              #   pca_space_ITD,pca_f_ITD,pca_coeff_ITD,pca_expVar_ITD, 
              #   pca_space_IAC_nf,pca_f_IAC_nf,pca_coeffs_IAC_nf,
              #   pca_expVar_IAC_nf, pca_space_ITD_nf,pca_f_ITD_nf,
              #   pca_coeffs_ITD_nf,pca_expVar_ITD_nf, ica_space_ITD,
              #   ica_f_ITD,ica_coeff_ITD, ica_space_IAC, ica_f_IAC, 
              #   ica_coeff_IAC, ica_space_ITD_nf,ica_f_ITD_nf,ica_coeffs_ITD_nf,
              #   ica_space_IAC_nf, ica_f_IAC_nf,ica_coeffs_IAC_nf] = pickle.load(file)

    A_IAC_Ht.append(IAC_Ht)
    A_ITD_Ht.append(ITD_Ht)
    A_IAC_Ht_nf.append(IAC_Htnf[:10])
    A_ITD_Ht_nf.append(ITD_Htnf[:10])
    
             
    
    
#%% Plot time domain
#get Ht into numpy vector

t2 = np.concatenate((-t[-1:0:-1],t))
    
Anp_Ht_IAC = np.zeros([A_IAC_Ht[0].shape[0],A_IAC_Ht[0].shape[1],len(Subjects)])
Anp_Ht_ITD = np.zeros([A_ITD_Ht[0].shape[0],A_ITD_Ht[0].shape[1],len(Subjects)])
for s in range(len(Subjects)):
    Anp_Ht_IAC[:,:,s] = A_IAC_Ht[s]
    Anp_Ht_ITD[:,:,s] = A_ITD_Ht[s]
    
    
Ht_avg_IAC = Anp_Ht_IAC.mean(axis=2)
Ht_avg_ITD = Anp_Ht_ITD.mean(axis=2)

sbp = [4,4]
sbp2 = [4,4]

fig,axs = plt.subplots(sbp[0],sbp[1],sharex=True)
for p1 in range(sbp[0]):
    for p2 in range(sbp[1]):
        for s in range(len(Subjects)):
            axs[p1,p2].plot(t2,Anp_Ht_IAC[p1*sbp[1]+p2,:,s])
            for n in range(len(A_IAC_Ht_nf[s])):
                axs[p1,p2].plot(t2,A_IAC_Ht_nf[s][n][p1*sbp[1]+p2,:],color='grey',alpha=0.3)
        axs[p1,p2].plot(t2,Ht_avg_IAC[p1*sbp[1]+p2,:],color='black',linewidth=2)            
        axs[p1,p2].set_title(ch_picks[p1*sbp[1]+p2])    

fig.suptitle('Ht IAC')

fig,axs = plt.subplots(sbp2[0],sbp2[1],sharex=True)
for p1 in range(sbp2[0]):
    for p2 in range(sbp2[1]):
        for s in range(len(Subjects)):
            axs[p1,p2].plot(t2,Anp_Ht_IAC[p1*sbp2[1]+p2+sbp[0]*sbp[1],:,s])
            for n in range(len(A_IAC_Ht_nf[s])):
                axs[p1,p2].plot(t2,A_IAC_Ht_nf[s][n][p1*sbp2[1]+p2+sbp[0]*sbp[1],:],color='grey',alpha=0.3)
        axs[p1,p2].plot(t2,Ht_avg_IAC[p1*sbp2[1]+p2+sbp[0]*sbp[1],:],color='black',linewidth=2)            
        axs[p1,p2].set_title(ch_picks[p1*sbp2[1]+p2+sbp[0]*sbp[1]])   
         
fig.suptitle('Ht IAC')


fig,axs = plt.subplots(sbp[0],sbp[1],sharex=True)
for p1 in range(sbp[0]):
    for p2 in range(sbp[1]):
        for s in range(len(Subjects)):
            axs[p1,p2].plot(t2,Anp_Ht_ITD[p1*sbp[1]+p2,:,s])
            for n in range(len(A_ITD_Ht_nf[s])):
                axs[p1,p2].plot(t2,A_ITD_Ht_nf[s][n][p1*sbp[1]+p2,:],color='grey',alpha=0.3)
        axs[p1,p2].plot(t2,Ht_avg_ITD[p1*sbp[1]+p2,:],color='black',linewidth=2)            
        axs[p1,p2].set_title(ch_picks[p1*sbp[1]+p2])    

fig.suptitle('Ht ITD')

fig,axs = plt.subplots(sbp2[0],sbp2[1],sharex=True)
for p1 in range(sbp2[0]):
    for p2 in range(sbp2[1]):
        for s in range(len(Subjects)):
            axs[p1,p2].plot(t2,Anp_Ht_ITD[p1*sbp2[1]+p2+sbp[0]*sbp[1],:,s])
            for n in range(len(A_ITD_Ht_nf[s])):
                axs[p1,p2].plot(t2,A_ITD_Ht_nf[s][n][p1*sbp2[1]+p2+sbp[0]*sbp[1],:],color='grey',alpha=0.3)
        axs[p1,p2].plot(t2,Ht_avg_ITD[p1*sbp2[1]+p2+sbp[0]*sbp[1],:],color='black',linewidth=2)            
        axs[p1,p2].set_title(ch_picks[p1*sbp2[1]+p2+sbp[0]*sbp[1]])   
         
fig.suptitle('Ht ITD')

    

#%% Get response from pca on average response

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

channels = [30,31,4,25,3,25]


if pca_Htavg_IACcoeffs[0,channels].mean() < pca_Htavg_IACcoeffs[0,:].mean():
    pca_Htavg_IACcoeffs = - pca_Htavg_IACcoeffs
    pca_sp_Htavg_IAC = - pca_sp_Htavg_IAC
    
if pca_Htavg_ITDcoeffs[0,channels].mean() < pca_Htavg_ITDcoeffs[0,:].mean():
    pca_Htavg_ITDcoeffs = - pca_Htavg_ITDcoeffs
    pca_sp_Htavg_ITD = - pca_sp_Htavg_ITD


vmin = pca_Htavg_IACcoeffs.mean() - 2*pca_Htavg_IACcoeffs.std()
vmax = pca_Htavg_IACcoeffs.mean() + 2*pca_Htavg_IACcoeffs.std()
plt.figure()
mne.viz.plot_topomap(pca_Htavg_IACcoeffs.squeeze(), info_obj,vmin=vmin,vmax=vmax)
plt.title('IAC: Avg Subjects then calculate coeffs')

# vmin = pca_Htavg_ITDcoeffs.mean() - 2*pca_Htavg_ITDcoeffs.std()
# vmax = pca_Htavg_ITDcoeffs.mean() - 2*pca_Htith open(os.path.join(data_loc, Subject+'_DynBin_SysFunc.pickle'),'rb') as file:     


plt.figure()
mne.viz.plot_topomap(pca_Htavg_ITDcoeffs.squeeze(), mne.pick_info(info_obj, ch_picks),vmin=vmin,vmax=vmax)
plt.title('ITD: Avg Subjects then calculate coeffs')


plt.figure()
plt.plot(t2, pca_sp_Htavg_IAC)

plt.figure()
plt.plot(t2, pca_sp_Htavg_ITD)


#%% Leave one out - jacknife

pca_sp_Htavg_IAC_JN = np.zeros([pca_sp_Htavg_IAC.shape[0],len(Subjects)])
pca_sp_Htavg_ITD_JN = np.zeros([pca_sp_Htavg_ITD.shape[0],len(Subjects)])
snums = np.arange(len(Subjects))
pca = PCA(n_components=1)
for jn in range(len(Subjects)):
    s_jn = np.delete(snums,jn)
    Ht_avg_IAC_JN = Anp_Ht_IAC[:,:,s_jn].mean(axis=2)
    Ht_avg_ITD_JN = Anp_Ht_ITD[:,:,s_jn].mean(axis=2)
    pca_IAC_JN = pca.fit_transform(Ht_avg_IAC_JN.T)[:,0]
    pca_IAC_coeffs = pca.components_
    if pca_IAC_coeffs[0,channels].mean() < pca_IAC_coeffs[0,:].mean():
        pca_IAC_coeffs = - pca_IAC_coeffs
        pca_IAC_JN = - pca_IAC_JN
        
    pca_ITD_JN = pca.fit_transform(Ht_avg_ITD_JN.T)[:,0]
    pca_ITD_coeffs = pca.components_
    if pca_ITD_coeffs[0,channels].mean() < pca_ITD_coeffs[0,:].mean():
        pca_ITD_coeffs = - pca_ITD_coeffs
        pca_ITD_JN = - pca_ITD_JN
    
    pca_sp_Htavg_IAC_JN[:,jn] = pca_IAC_JN
    pca_sp_Htavg_ITD_JN[:,jn] = pca_ITD_JN
    
IAC_pcaJN_se = np.sqrt( (pca_sp_Htavg_IAC_JN.shape[1]-1) * np.sum( (pca_sp_Htavg_IAC_JN - pca_sp_Htavg_IAC_JN.mean(axis=1)[:,np.newaxis]) **2,axis=1 ) / pca_sp_Htavg_IAC_JN.shape[1]  )
ITD_pcaJN_se = np.sqrt( (pca_sp_Htavg_ITD_JN.shape[1]-1) * np.sum( (pca_sp_Htavg_ITD_JN - pca_sp_Htavg_ITD_JN.mean(axis=1)[:,np.newaxis]) **2,axis=1 ) / pca_sp_Htavg_ITD_JN.shape[1] )

plt.figure()
plt.plot(t2, pca_sp_Htavg_IAC,color='k')
plt.fill_between(t2,pca_sp_Htavg_IAC[:,0]-2*IAC_pcaJN_se,pca_sp_Htavg_IAC[:,0]+2*IAC_pcaJN_se)


plt.figure()
plt.plot(t2, pca_sp_Htavg_ITD,color='k')
plt.fill_between(t2,pca_sp_Htavg_ITD[:,0]-2*ITD_pcaJN_se,pca_sp_Htavg_ITD[:,0]+2*ITD_pcaJN_se)

#%% Noise Floor computation 

Anp_IAC_Ht_nf = np.zeros([A_IAC_Ht_nf[0][0].shape[0],A_IAC_Ht_nf[0][0].shape[1],len(A_IAC_Ht_nf[0]*len(Subjects))])
for s in range(len(Subjects)):
    for nf in range(len(A_IAC_Ht_nf[0])):
        Anp_IAC_Ht_nf[:,:,s*nf+nf] = A_IAC_Ht_nf[s][nf]
        
pca_nf_HTavg_IAC_JN = np.zeros([pca_sp_Htavg_IAC.shape[0],Anp_IAC_Ht_nf.shape[2]])
pca_IACnf_comp = np.zeros([32,Anp_IAC_Ht_nf.shape[2]])
for jn in range(Anp_IAC_Ht_nf.shape[2]):
    print('On nf JN: ' + str(jn))
    s_jn = np.delete(np.arange(Anp_IAC_Ht_nf.shape[2]),jn)
    Htnf_avg_IAC_JN = Anp_IAC_Ht_nf[:,:,s_jn].mean(axis=2)
    pca_IACnf_JN = pca.fit_transform(Htnf_avg_IAC_JN.T)[:,0]
    pca_IACnf_coeffs = pca.components_
    if pca_IACnf_coeffs[0,channels].mean() < pca_IACnf_coeffs[0,:].mean():
        pca_IACnf_coeffs = - pca_IACnf_coeffs
        pca_IACnf_JN = - pca_IACnf_JN
    pca_IACnf_comp[:,jn] = pca_IACnf_coeffs.squeeze()
    pca_nf_HTavg_IAC_JN[:,jn] = pca_IACnf_JN
    
IACnf_pcaJN_se = np.sqrt( (pca_nf_HTavg_IAC_JN.shape[1]-1) * np.sum( (pca_nf_HTavg_IAC_JN - pca_nf_HTavg_IAC_JN.mean(axis=1)[:,np.newaxis]) **2,axis=1 ) / pca_sp_Htavg_IAC_JN.shape[1]  )

plt.figure()
plt.plot(t2,pca_nf_HTavg_IAC_JN.mean(axis=1),color='grey')
plt.fill_between(t2,pca_nf_HTavg_IAC_JN.mean(axis=1)-2*IACnf_pcaJN_se,pca_nf_HTavg_IAC_JN.mean(axis=1)+2*IACnf_pcaJN_se,color='grey',alpha=0.3)
plt.plot(t2, pca_sp_Htavg_IAC,color='k')
plt.fill_between(t2,pca_sp_Htavg_IAC[:,0]-2*ITD_pcaJN_se,pca_sp_Htavg_IAC[:,0]+2*ITD_pcaJN_se)


plt.figure()
mne.viz.plot_topomap(pca_IACnf_comp.mean(axis=1), info_obj,vmin=vmin,vmax=vmax)
plt.title('IAC: Noise PCA coeffs')



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







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
from scipy.special import erf
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA


#data_loc = os.path.abspath('/media/ravinderjit/Data_Drive/Data/EEGdata/DynamicBinaural/Pickles_32/SystemFuncs32_2')
data_loc = os.path.abspath('/media/ravinderjit/Data_Drive/Data/EEGdata/DynamicBinaural/Pickles_32_refAvg/')
fig_path = os.path.abspath('/media/ravinderjit/Data_Drive/Data/Figures/DynBin/')


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
        [t, IAC_Ht, ITD_Ht, IAC_Htnf, ITD_Htnf,Tot_trials_IAC,Tot_trials_ITD] = pickle.load(file)
        
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
            axs[p1,p2].plot(t,Anp_Ht_IAC[p1*sbp[1]+p2,:,s])
            for n in range(len(A_IAC_Ht_nf[s])):
                axs[p1,p2].plot(t,A_IAC_Ht_nf[s][n][p1*sbp[1]+p2,:],color='grey',alpha=0.3)
        axs[p1,p2].plot(t,Ht_avg_IAC[p1*sbp[1]+p2,:],color='black',linewidth=2)            
        axs[p1,p2].set_title(ch_picks[p1*sbp[1]+p2])    

fig.suptitle('Ht IAC')

fig,axs = plt.subplots(sbp2[0],sbp2[1],sharex=True)
for p1 in range(sbp2[0]):
    for p2 in range(sbp2[1]):
        for s in range(len(Subjects)):
            axs[p1,p2].plot(t,Anp_Ht_IAC[p1*sbp2[1]+p2+sbp[0]*sbp[1],:,s])
            for n in range(len(A_IAC_Ht_nf[s])):
                axs[p1,p2].plot(t,A_IAC_Ht_nf[s][n][p1*sbp2[1]+p2+sbp[0]*sbp[1],:],color='grey',alpha=0.3)
        axs[p1,p2].plot(t,Ht_avg_IAC[p1*sbp2[1]+p2+sbp[0]*sbp[1],:],color='black',linewidth=2)            
        axs[p1,p2].set_title(ch_picks[p1*sbp2[1]+p2+sbp[0]*sbp[1]])   
         
fig.suptitle('Ht IAC')


fig,axs = plt.subplots(sbp[0],sbp[1],sharex=True)
for p1 in range(sbp[0]):
    for p2 in range(sbp[1]):
        for s in range(len(Subjects)):
            axs[p1,p2].plot(t,Anp_Ht_ITD[p1*sbp[1]+p2,:,s])
            for n in range(len(A_ITD_Ht_nf[s])):
                axs[p1,p2].plot(t,A_ITD_Ht_nf[s][n][p1*sbp[1]+p2,:],color='grey',alpha=0.3)
        axs[p1,p2].plot(t,Ht_avg_ITD[p1*sbp[1]+p2,:],color='black',linewidth=2)            
        axs[p1,p2].set_title(ch_picks[p1*sbp[1]+p2])    

fig.suptitle('Ht ITD')

fig,axs = plt.subplots(sbp2[0],sbp2[1],sharex=True)
for p1 in range(sbp2[0]):
    for p2 in range(sbp2[1]):
        for s in range(len(Subjects)):
            axs[p1,p2].plot(t,Anp_Ht_ITD[p1*sbp2[1]+p2+sbp[0]*sbp[1],:,s])
            for n in range(len(A_ITD_Ht_nf[s])):
                axs[p1,p2].plot(t,A_ITD_Ht_nf[s][n][p1*sbp2[1]+p2+sbp[0]*sbp[1],:],color='grey',alpha=0.3)
        axs[p1,p2].plot(t,Ht_avg_ITD[p1*sbp2[1]+p2+sbp[0]*sbp[1],:],color='black',linewidth=2)            
        axs[p1,p2].set_title(ch_picks[p1*sbp2[1]+p2+sbp[0]*sbp[1]])   
         
fig.suptitle('Ht ITD')


#%% Get response from pca on average response

t_1 = np.where(t>=-1.25)[0][0]
t_2 = np.where(t>=-0.75)[0][0]

t_1 = np.where(t>=0)[0][0]
t_2 = np.where(t>=0.5)[0][0]


pca = PCA(n_components=1)
pca_sp_Htavg_IAC = pca.fit_transform(Ht_avg_IAC[:,t_1:t_2].T)
pca_Htavg_IACcoeffs = pca.components_
pca_Htavg_IACexpVar = pca.explained_variance_ratio_

pca = PCA(n_components=1)
pca_sp_Htavg_ITD = pca.fit_transform(Ht_avg_ITD[:,t_1:t_2].T)
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
plt.savefig(os.path.join(fig_path, 'IACavg_PCA_topomap.svg') , format='svg')
plt.title('IAC: Avg Subjects then calculate coeffs')

# vmin = pca_Htavg_ITDcoeffs.mean() - 2*pca_Htavg_ITDcoeffs.std()
# vmax = pca_Htavg_ITDcoeffs.mean() - 2*pca_Htith open(os.path.join(data_loc, Subject+'_DynBin_SysFunc.pickle'),'rb') as file:     


plt.figure()
mne.viz.plot_topomap(pca_Htavg_ITDcoeffs.squeeze(), mne.pick_info(info_obj, ch_picks),vmin=vmin,vmax=vmax)
plt.savefig(os.path.join(fig_path, 'ITDavg_PCA_topomap.svg') , format='svg')
plt.title('ITD: Avg Subjects then calculate coeffs')


plt.figure()
plt.plot(t[t_1:t_2], pca_sp_Htavg_IAC)
plt.title('IAC')

plt.figure()
plt.plot(t[t_1:t_2], pca_sp_Htavg_ITD)
plt.title('ITD')

#%% Leave one out - jacknife

pca_sp_Htavg_IAC_JN = np.zeros([pca_sp_Htavg_IAC.shape[0],len(Subjects)])
pca_sp_Htavg_ITD_JN = np.zeros([pca_sp_Htavg_ITD.shape[0],len(Subjects)])

pca_IAC_comp = np.zeros([32,Anp_Ht_IAC.shape[2]])

snums = np.arange(len(Subjects))
pca = PCA(n_components=1)
for jn in range(len(Subjects)):
    s_jn = np.delete(snums,jn)
    Ht_avg_IAC_JN = Anp_Ht_IAC[:,t_1:t_2,s_jn].mean(axis=2)
    Ht_avg_ITD_JN = Anp_Ht_ITD[:,t_1:t_2,s_jn].mean(axis=2)
    pca_IAC_JN = pca.fit_transform(Ht_avg_IAC_JN.T)[:,0]
    pca_IAC_coeffs = pca.components_
    
    
    if pca_IAC_coeffs[0,channels].mean() < pca_IAC_coeffs[0,:].mean():
        pca_IAC_coeffs = - pca_IAC_coeffs
        pca_IAC_JN = - pca_IAC_JN
        
    if pca_IAC_coeffs[0,channels].mean() < 0: ## Make center channels positive if negative
        pca_IAC_coeffs = - pca_IAC_coeffs
        pca_IAC_JN = - pca_IAC_JN
        
    pca_ITD_JN = pca.fit_transform(Ht_avg_ITD_JN.T)[:,0]
    pca_ITD_coeffs = pca.components_
    
    
    if pca_ITD_coeffs[0,channels].mean() < pca_ITD_coeffs[0,:].mean():
        pca_ITD_coeffs = - pca_ITD_coeffs
        pca_ITD_JN = - pca_ITD_JN
        
    if pca_ITD_coeffs[0,channels].mean() < 0: ## Make center channels positive if negative
        pca_ITD_coeffs = - pca_ITD_coeffs
        pca_ITD_JN = - pca_ITD_JN
    
    pca_IAC_comp[:,jn] = pca_IAC_coeffs.squeeze()
    
    pca_sp_Htavg_IAC_JN[:,jn] = pca_IAC_JN
    pca_sp_Htavg_ITD_JN[:,jn] = pca_ITD_JN
    
IAC_pcaJN_se = np.sqrt( (pca_sp_Htavg_IAC_JN.shape[1]-1) * np.sum( (pca_sp_Htavg_IAC_JN - pca_sp_Htavg_IAC_JN.mean(axis=1)[:,np.newaxis]) **2,axis=1 ) / pca_sp_Htavg_IAC_JN.shape[1]  )
ITD_pcaJN_se = np.sqrt( (pca_sp_Htavg_ITD_JN.shape[1]-1) * np.sum( (pca_sp_Htavg_ITD_JN - pca_sp_Htavg_ITD_JN.mean(axis=1)[:,np.newaxis]) **2,axis=1 ) / pca_sp_Htavg_ITD_JN.shape[1] )

plt.figure()
plt.plot(t[t_1:t_2], pca_sp_Htavg_IAC,color='k')
plt.fill_between(t[t_1:t_2],pca_sp_Htavg_IAC[:,0]-2*IAC_pcaJN_se,pca_sp_Htavg_IAC[:,0]+2*IAC_pcaJN_se)


plt.figure()
plt.plot(t[t_1:t_2], pca_sp_Htavg_ITD,color='k')
plt.fill_between(t[t_1:t_2],pca_sp_Htavg_ITD[:,0]-2*ITD_pcaJN_se,pca_sp_Htavg_ITD[:,0]+2*ITD_pcaJN_se)


plt.figure()
mne.viz.plot_topomap(pca_IAC_comp.mean(axis=1), info_obj,vmin=vmin,vmax=vmax)
plt.title('IAC: PCA coeffs mean JN')


#%% Compute Noise Floors By applying real topomap coeffs to NFs

All_IACnfs = np.zeros([t_2-t_1, len(Subjects)*len(A_IAC_Ht_nf[0])])
All_ITDnfs = np.zeros([t_2-t_1, len(Subjects)*len(A_ITD_Ht_nf[0])])
it = 0
for sub in range(len(Subjects)):
    for nf in range(len(A_IAC_Ht_nf[0])):
        _IACnf = A_IAC_Ht_nf[sub][nf][:,t_1:t_2] - A_IAC_Ht_nf[sub][nf][:,t_1:t_2].mean(axis=1)[:,np.newaxis]
        _ITDnf = A_ITD_Ht_nf[sub][nf][:,t_1:t_2] - A_ITD_Ht_nf[sub][nf][:,t_1:t_2].mean(axis=1)[:,np.newaxis]
        
        _IACnf = np.matmul(pca_Htavg_IACcoeffs, _IACnf)
        _ITDnf = np.matmul(pca_Htavg_ITDcoeffs, _ITDnf)
        All_IACnfs[:,it] = _IACnf
        All_ITDnfs[:,it] = _ITDnf
        it+=1
        

IACnf_se = All_IACnfs.std(axis=1) / np.sqrt(len(Subjects))
ITDnf_se = All_ITDnfs.std(axis=1) / np.sqrt(len(Subjects))

plt.figure()
plt.plot(t[t_1:t_2],All_IACnfs.mean(axis=1),color='grey')
plt.fill_between(t[t_1:t_2],All_IACnfs.mean(axis=1) - 2*IACnf_se, All_IACnfs.mean(axis=1) + 2*IACnf_se,color='grey')
plt.plot(t[t_1:t_2], pca_sp_Htavg_IAC,color='k')
plt.fill_between(t[t_1:t_2],pca_sp_Htavg_IAC[:,0]-2*IAC_pcaJN_se,pca_sp_Htavg_IAC[:,0]+2*IAC_pcaJN_se,color='k')


plt.figure()
plt.plot(t[t_1:t_2],All_ITDnfs.mean(axis=1),color='grey')
plt.fill_between(t[t_1:t_2],All_ITDnfs.mean(axis=1) - 2*ITDnf_se, All_ITDnfs.mean(axis=1) + 2*ITDnf_se,color='grey')
plt.plot(t[t_1:t_2], pca_sp_Htavg_ITD,color='k')
plt.fill_between(t[t_1:t_2],pca_sp_Htavg_ITD[:,0]-2*ITD_pcaJN_se,pca_sp_Htavg_ITD[:,0]+2*ITD_pcaJN_se,color='k')
    

#%% Behavioral Binaural Unmasking Dynamics

def expFit(x,A,tau):
    Out = A*(1 - np.exp(-x/tau))
    return Out

stDevNeg = 0.1
stDevPos = 0.3
tneg = np.arange(-stDevNeg*5,0,.01)
tpos = np.arange(0,stDevPos*5+.01,.01)

simpGaussNeg = np.exp(-0.5*(tneg/stDevNeg)**2)
simpGaussPos = np.exp(-0.5*(tpos/stDevPos)**2)

simpGauss = np.concatenate((simpGaussNeg,simpGaussPos))
t = np.concatenate((tneg,tpos))

plt.figure()
plt.plot(t,simpGauss)

beh_dataPath = '/media/ravinderjit/Data_Drive/Data/BehaviorData/IACbehavior/'
beh_IACsq = loadmat(os.path.abspath(beh_dataPath+'IACsquareTone_Processed.mat'))

Window_t = beh_IACsq['WindowSizes'][0]
f_beh = Window_t[1:]**-1
SNRs_beh = beh_IACsq['AcrossSubjectsSNR'] - beh_IACsq['AcrossSubjectsSNR'][0]
AcrossSubjectSEM = beh_IACsq['AcrossSubjectsSEM']

# SNRs_beh = 10**(SNRs_beh/20)

plt.figure()
plt.errorbar(Window_t,SNRs_beh,AcrossSubjectSEM)

# popt, pcov = sp.optimize.curve_fit(expFit, Window_t,SNRs_beh[:,0],bounds=([0,-np.inf],[np.inf,np.inf]),p0=[10,0.1])
# A = popt[0]
# tau = popt[1]
# fit_t = np.arange(Window_t[0],Window_t[-1],1/fs)
# exp_fit = expFit(fit_t,A,tau)


Tnu = 10**(-beh_IACsq['AcrossSubjectsSNR'][0]/10)
Tno = 10**(-beh_IACsq['AcrossSubjectsSNR'][9]/10)

rho = np.arange(0,1.01,.01)
eq = -10*np.log10(rho + (1-rho)*(Tnu/Tno))

#eq2 = -10*np.log10(0.5*(1+rho) + 0.5*(1-rho) * (Tnu/Tno))

plt.figure()
plt.plot(rho,eq)
#plt.plot(rho,eq2)


plt.figure()
plt.errorbar(Window_t,SNRs_beh,AcrossSubjectSEM)
# plt.plot(fit_t,exp_fit)


# w,h_behFit = freqz(exp_fit,a=1,worN=2000,fs=fs)

# plt.figure()
# plt.plot(w,np.abs(20*np.log10(h_behFit)))
# plt.xlim([0,20])

plt.figure()
plt.errorbar(f_beh/2,SNRs_beh[1:,0],AcrossSubjectSEM[1:,0])




#%% Frequency domain 
      
b_IAC = pca_sp_Htavg_IAC
#b = b[:int(np.round(.2*fs))]

w,h_IAC = freqz(b_IAC,a=1,worN=2000,fs=fs)

phase_IAC = np.unwrap(np.angle(h_IAC))


plt.figure()
plt.plot(w,np.abs(h_IAC))
plt.xlim([0,20])

f_2_ind = np.where(w>=2.5)[0][0]
f_5_ind = np.where(w>=6)[0][0]

coeff = np.polyfit(w[f_2_ind:f_5_ind],phase_IAC[f_2_ind:f_5_ind],deg=1)
GD_line = coeff[0] * w[f_2_ind:f_5_ind] + coeff[1]
GD_IAC = -coeff[0] / (2*np.pi)

plt.figure()
plt.plot(w,phase_IAC)
plt.plot(w[f_2_ind:f_5_ind],GD_line,color='k')
plt.xlim([0,20])
plt.ylim([-30,5])



b_ITD = pca_sp_Htavg_ITD
#b_ITD = b_ITD[:int(np.round(.2*fs))]

w_itd,h_itd = freqz(b_ITD,a=1,worN=2000,fs=fs)
phase_itd = np.unwrap(np.angle(h_itd))

plt.figure()
plt.plot(w_itd,np.abs(h_itd))
plt.xlim([0,20])

plt.figure()
plt.plot(w_itd,np.abs(h_itd))
plt.plot(w,np.abs(h_IAC))
plt.xlim([0,20])

plt.figure()
plt.plot(w,20*np.log10(np.abs(h_IAC)),label='IAC')
plt.plot(w_itd, 20*np.log10(np.abs(h_itd)),label='ITD')
plt.xlim([0,20])
plt.ylim([-40,5])
plt.legend()

f_2_ind = np.where(w_itd>=2.5)[0][0]
f_8_ind = np.where(w_itd>=6)[0][0]

coeff = np.polyfit(w_itd[f_2_ind:f_8_ind],phase_itd[f_2_ind:f_8_ind],deg=1)
GD_line = coeff[0] * w_itd[f_2_ind:f_8_ind] + coeff[1]
GD_ITD = -coeff[0] / (2*np.pi)

plt.figure()
plt.plot(w_itd,phase_itd)
plt.plot(w_itd[f_2_ind:f_8_ind],GD_line,color='black')
plt.xlim([0,20])
plt.ylim([-5,2])







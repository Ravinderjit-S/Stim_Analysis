#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 20:54:08 2021

@author: ravinderjit
This script averages mcBTRFs across subjects and does a PCA analysis to get a
sBTRF. Also comparison with behavior
"""

import os
import pickle
import numpy as np
import mne
import matplotlib.pyplot as plt
import matplotlib 
from scipy.signal import freqz
from scipy.io import loadmat
from sklearn.decomposition import PCA
from scipy.signal import hilbert
from scipy.io import savemat

data_loc = os.path.abspath('/media/ravinderjit/Data_Drive/Data/EEGdata/DynamicBinaural/Pickles_32_refAvg/')
fig_path = os.path.abspath('/media/ravinderjit/Data_Drive/Data/Figures/DynBin/')


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

    A_IAC_Ht.append(IAC_Ht)
    A_ITD_Ht.append(ITD_Ht)
    A_IAC_Ht_nf.append(IAC_Htnf)
    A_ITD_Ht_nf.append(ITD_Htnf)
    

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
        axs[p1,p2].set_xlim([-0.1,1])

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
        axs[p1,p2].set_xlim([-0.1,1])
         
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
        axs[p1,p2].set_xlim([-0.1,1])

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
        axs[p1,p2].set_xlim([-0.1,1])
         
fig.suptitle('Ht ITD')


#%% Get response from pca on average response

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

fontsz = 11
fig, ax = plt.subplots(1)
fig.set_size_inches(3.5,4)
dIAC =  np.diff(pca_sp_Htavg_IAC,axis=0)
ax.plot(t[t_1:t_2], pca_sp_Htavg_ITD/np.max(np.abs(pca_sp_Htavg_ITD)),color='k',linewidth=2)
ax.plot(t[t_1+1:t_2], dIAC/np.max(np.abs(dIAC)),color='grey',linewidth=2)
plt.legend(['ITD','dIAC'])
ax.set_ylabel('Normalized Amplitude')
ax.set_xlabel('Time (Sec)')
ax.set_yticks([-1, -0.5, 0, 0.5, 1])
ax.set_xticks([0, 0.25, 0.5])
ax.set_xlim([0,0.5])
ax.set_ylim([-1.1,1.1])
matplotlib.rcParams.update({'font.size':fontsz, 'font.family': 'sans-serif', 'font.sans-serif':['Arial']})
plt.savefig(os.path.join(fig_path, 'ITD_dIAC.svg') , format='svg')

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

#%% Make Time Domain sBTRF Figures

fontsz = 11

fig, ax = plt.subplots(1)
fig.set_size_inches(3.5,4)
ax.plot(t[t_1:t_2],All_IACnfs.mean(axis=1),color='grey',linewidth=2)
ax.fill_between(t[t_1:t_2],All_IACnfs.mean(axis=1) - 2*IACnf_se, All_IACnfs.mean(axis=1) + 2*IACnf_se,color='lightgrey')
ax.plot(t[t_1:t_2], pca_sp_Htavg_IAC,color='k',linewidth=2)
ax.fill_between(t[t_1:t_2],pca_sp_Htavg_IAC[:,0]-2*IAC_pcaJN_se,pca_sp_Htavg_IAC[:,0]+2*IAC_pcaJN_se,color=[0.25,0.25,0.25])
ax.set_xlim([0,0.5])
ax.set_yticks([-3e-3,0,3e-3,6e-3])
ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
ax.set_xlabel('Time (sec)')
matplotlib.rcParams.update({'font.size':fontsz, 'font.family': 'sans-serif', 'font.sans-serif':['Arial']})
plt.savefig(os.path.join(fig_path, 'IAC_HtAvg_pca.svg') , format='svg')

fig,ax = plt.subplots(1)
fig.set_size_inches(3.5,4)
ax.plot(t[t_1:t_2],All_ITDnfs.mean(axis=1),color='grey',linewidth=2)
ax.fill_between(t[t_1:t_2],All_ITDnfs.mean(axis=1) - 2*ITDnf_se, All_ITDnfs.mean(axis=1) + 2*ITDnf_se,color='lightgrey')
ax.plot(t[t_1:t_2], pca_sp_Htavg_ITD,color='k',linewidth=2)
ax.fill_between(t[t_1:t_2],pca_sp_Htavg_ITD[:,0]-2*ITD_pcaJN_se,pca_sp_Htavg_ITD[:,0]+2*ITD_pcaJN_se,color=[0.25,0.25,0.25])
ax.set_xlim([0,0.5])
ax.set_yticks([-4e-3,-2e-3,0,2e-3,4e-3])
ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
ax.set_xlabel('Time (sec)')
matplotlib.rcParams.update({'font.size':fontsz, 'font.family': 'sans-serif', 'font.sans-serif':['Arial']})
plt.savefig(os.path.join(fig_path, 'ITD_HtAvg_pca.svg') , format='svg')

#%% Frequency domain 
      
b_IAC = pca_sp_Htavg_IAC
w_IAC,h_IAC = freqz(b_IAC,a=1,worN=2000,fs=fs)
phase_IAC = np.unwrap(np.angle(h_IAC))

f_2_ind = np.where(w_IAC>=2.5)[0][0]
f_5_ind = np.where(w_IAC>=6)[0][0]

coeff = np.polyfit(w_IAC[f_2_ind:f_5_ind],phase_IAC[f_2_ind:f_5_ind],deg=1)
GD_line_IAC = coeff[0] * w_IAC[f_2_ind:f_5_ind] + coeff[1]
GD_IAC = -coeff[0] / (2*np.pi)

b_ITD = pca_sp_Htavg_ITD
w_itd,h_itd = freqz(b_ITD,a=1,worN=2000,fs=fs)
phase_itd = np.unwrap(np.angle(h_itd))

f_2_ind = np.where(w_itd>=2.5)[0][0]
f_8_ind = np.where(w_itd>=6)[0][0]

coeff = np.polyfit(w_itd[f_2_ind:f_8_ind],phase_itd[f_2_ind:f_8_ind],deg=1)
GD_line_ITD = coeff[0] * w_itd[f_2_ind:f_8_ind] + coeff[1]
GD_ITD = -coeff[0] / (2*np.pi)

h_IAC_noise = np.zeros([2000,All_IACnfs.shape[1]])
for nf in range(All_IACnfs.shape[1]):
    w_IAC_noise, h_IAC_noise[:,nf] = np.abs(freqz(All_IACnfs[:,nf],a=1,worN=2000,fs=fs))
    
h_ITD_noise = np.zeros([2000,All_ITDnfs.shape[1]])
for nf in range(All_ITDnfs.shape[1]):
    w_ITD_noise, h_ITD_noise[:,nf] = np.abs(freqz(All_ITDnfs[:,nf],a=1,worN=2000,fs=fs))
    
h_IAC_jn = np.zeros([2000,pca_sp_Htavg_IAC_JN.shape[1]])
for jn in range(pca_sp_Htavg_IAC_JN.shape[1]):
    w_IAC_jn, h_IAC_jn[:,jn] = np.abs(freqz(pca_sp_Htavg_IAC_JN[:,jn],a=1,worN=2000,fs=fs))
    
h_ITD_jn = np.zeros([2000,pca_sp_Htavg_ITD_JN.shape[1]])
for jn in range(pca_sp_Htavg_ITD_JN.shape[1]):
    w_ITD_jn, h_ITD_jn[:,jn] = np.abs(freqz(pca_sp_Htavg_ITD_JN[:,jn],a=1,worN=2000,fs=fs))

h_IAC_pcaJN_se = np.sqrt( (h_IAC_jn.shape[1]-1) * np.sum( (h_IAC_jn - h_IAC_jn.mean(axis=1)[:,np.newaxis]) **2,axis=1 ) / h_IAC_jn.shape[1]  )
h_ITD_pcaJN_se = np.sqrt( (h_ITD_jn.shape[1]-1) * np.sum( (h_ITD_jn - h_ITD_jn.mean(axis=1)[:,np.newaxis]) **2,axis=1 ) / h_ITD_jn.shape[1]  )
    

h_IAC_noise_se = h_IAC_noise.std(axis=1) / np.sqrt(len(Subjects))
h_ITD_noise_se = h_ITD_noise.std(axis=1) / np.sqrt(len(Subjects))

fig,ax = plt.subplots(1)
fig.set_size_inches(3.5,4)
ax2 = ax.twinx()
p1, = ax.plot(w_IAC,np.abs(h_IAC),color='k',linewidth=2,label='Magnitude')
ax.fill_between(w_IAC,np.abs(h_IAC) - 2*h_IAC_pcaJN_se, np.abs(h_IAC) + 2*h_IAC_pcaJN_se,color=[0.25,0.25,0.25] )
p2, = ax2.plot(w_IAC,phase_IAC,linestyle='--',linewidth=2,color='k',label='Phase')
p3, = ax.plot(w_IAC_noise,h_IAC_noise.mean(axis=1),color='grey',label='NoiseFloor')
ax.fill_between(w_IAC_noise,h_IAC_noise.mean(axis=1) - 2 * h_IAC_noise_se, h_IAC_noise.mean(axis=1) + 2*h_IAC_noise_se, color='lightgrey',alpha=0.7)
ax.set_xlim([0,10])
ax.set_yticks([0,0.4,0.8,1.2,1.6])
ax.set_ylim([0,2.0])
ax2.set_ylim([-10,1])
ax2.set_yticks([0,-3,-6,-9])
#ax.set_ylabel('Mag')
#ax2.set_ylabel('Phase (Rad)')
ax.set_xlabel('Frequency (Hz)')
lines = [p1,p3, p2]
ax.legend(lines,[l.get_label() for l in lines])
matplotlib.rcParams.update({'font.size':fontsz, 'font.family': 'sans-serif', 'font.sans-serif':['Arial']})
plt.savefig(os.path.join(fig_path, 'IAC_HfAvg_pca.svg') , format='svg')


fig,ax = plt.subplots(1)
fig.set_size_inches(3.5,4)
ax2 = ax.twinx()
p1, = ax.plot(w_itd,np.abs(h_itd),color='k',linewidth=2,label='Magnitude')
ax.fill_between(w_itd,np.abs(h_itd) - 2*h_ITD_pcaJN_se, np.abs(h_itd) + 2*h_ITD_pcaJN_se,color=[0.25,0.25,0.25] )
p2, = ax2.plot(w_itd,phase_itd,linestyle='--',linewidth=2,color='k',label='Phase')
p3, = ax.plot(w_ITD_noise,h_ITD_noise.mean(axis=1),color='grey',label='NoiseFloor')
ax.fill_between(w_ITD_noise,h_ITD_noise.mean(axis=1) - 2 *h_ITD_noise_se, h_ITD_noise.mean(axis=1) + 2*h_ITD_noise_se, color='lightgrey',alpha=0.7)
ax.set_xlim([0,10])
ax.set_ylim([0,0.85])
ax.set_yticks([0,0.2,0.4,0.6,0.8])
ax2.set_ylim([-6,2])
ax2.set_yticks([1,-1,-3,-5])
#ax.set_ylabel('Mag')
#ax2.set_ylabel('Phase (Rad)')
ax.set_xlabel('Frequency (Hz)')
lines = [p1,p3, p2]
#ax.legend(lines,[l.get_label() for l in lines])
matplotlib.rcParams.update({'font.size':fontsz, 'font.family': 'sans-serif', 'font.sans-serif':['Arial']})
plt.savefig(os.path.join(fig_path, 'ITD_HfAvg_pca.svg') , format='svg')

with open('_DynBin_SysFunc_PCA_Ht' + '.pickle','wb') as file:     
    pickle.dump([pca_sp_Htavg_IAC],file)


#%% Compare Binaural Behavior with Physiology
beh_dataPath = '/media/ravinderjit/Data_Drive/Data/BehaviorData/IACbehavior/'
beh_IACsq = loadmat(os.path.abspath(beh_dataPath+'IACsquareTone_Processed.mat'))

Window_t = beh_IACsq['WindowSizes'][0]
f_beh = Window_t[1:]**-1
SNRs_beh = beh_IACsq['AcrossSubjectsSNR']
AcrossSubjectSEM = beh_IACsq['AcrossSubjectsSEM']

physWind = pca_sp_Htavg_IAC

t_phys = np.arange(0,0.5,1/fs)
physWind = physWind
physWind = physWind - physWind[0]
zero_crossings = np.where(np.diff(np.sign(physWind[:,0])))[0]
t1 = np.where(t_phys>=0.450)[0][0] #mean of response touches mean of NF around 440 ms
t_phys = t_phys[:t1] 
physWind = physWind[:t1]

env = np.abs(hilbert(physWind))
env = env / np.sum(env)
physWind = physWind - min(physWind)
physWind = physWind / np.sum(physWind)

plt.figure()
plt.plot(t_phys,env)
plt.plot(t_phys,physWind)

#physWind = env

winds = Window_t
out = np.zeros(winds.size)
plt.figure()

out_convolves = []
maxIAC_overlap = np.zeros(winds.size)

Tnu = 10**(-beh_IACsq['AcrossSubjectsSNR'][0]/10)
Tno = 10**(-beh_IACsq['AcrossSubjectsSNR'][9]/10) 
rho = np.arange(0,1.01,.01)
eq = -10*np.log10((1-rho) + (rho)*(Tno/Tnu))   

for j in range(0,winds.size):
    wind_j = winds[j]
    
    IAC_0block = np.ones([int(np.round(fs*0.4))]) * 0
    IAC_1block = np.ones([int(np.round(fs*wind_j))]) * 1
    IACnoise = np.concatenate((IAC_0block,IAC_1block,IAC_0block))
    
    out_conv = np.convolve(IACnoise,physWind[:,0],mode='full')
    t_xcorr = np.arange(-(physWind.size-1)/fs,(out_conv.size-physWind.size+1)/fs,1/fs)
    
    #where does window overlap with tone
    t_overlapStart = np.where(t_xcorr+physWind.size/fs > (IACnoise.size/2+ 0.01*fs)/fs)[0][0]
    t_overlapEnd = np.where(t_xcorr-physWind.size/fs < (IACnoise.size/2 - 0.01*fs)/fs)[0][-1]
    
    out_convolves.append(out_conv)
    
    maxIAC = np.max(out_conv[t_overlapStart:t_overlapEnd])
    maxIAC_overlap[j] = maxIAC
    
    if(maxIAC >=1):
        ind_rho = -1
    elif(maxIAC <= 0):
        ind_rho = 0
    else:
        ind_rho = np.where(rho>=maxIAC)[0][0]
        
    out[j] = eq[ind_rho]
        
    plt.plot(t_xcorr,out_conv,label=j)
    plt.scatter((IACnoise.size/2+.01*fs)/fs,0,color='red')
    plt.scatter(t_xcorr[t_overlapStart],np.zeros(t_overlapStart.size),color='k')
    plt.legend()
    
 
BMLD = SNRs_beh[:,0] - SNRs_beh[0]
mse_physFit = np.mean((BMLD[1:9] - out[1:9])**2)
    
fig, ax = plt.subplots(1)
fig.set_size_inches(3.5,4)
ax.errorbar(Window_t,BMLD, AcrossSubjectSEM, label='behavior',color='k',fmt='o',linewidth=2) #behavior
ax.plot(Window_t,out, label='physio')
ax.scatter(Window_t,out)
ax.set_xlabel('Time (sec)')
ax.set_ylabel('Detection Improvement (dB)')
#plt.scatter(Window_t,out)
plt.legend()

#Gonna save varible to make figure in matlab since other behavioral data is analyzed there
savemat('Physio_behModel.mat',{'PhysBehMod': out})

fontsz = 11
fig,ax = plt.subplots(1)
fig.set_size_inches(3.3,3.3)
ax.plot(t_phys,physWind,color='k',linewidth=2)
ax.set_yticks([0,.003])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
ax.set_xlabel('Time(sec)')
matplotlib.rcParams.update({'font.size':fontsz, 'font.family': 'sans-serif', 'font.sans-serif':['Arial']})
plt.savefig(os.path.join(fig_path, 'IAC_Ht_behNorm.svg') , format='svg')

fig,ax = plt.subplots(1)
fig.set_size_inches(3.3,3.3)
ax.plot(rho,eq,linewidth=2)
ax.set_xlabel('IAC')
ax.set_ylabel('BMLD (dB)')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
matplotlib.rcParams.update({'font.size':fontsz, 'font.family': 'sans-serif', 'font.sans-serif':['Arial']})
plt.savefig(os.path.join(fig_path, 'IAC_BMLD.svg') , format='svg')


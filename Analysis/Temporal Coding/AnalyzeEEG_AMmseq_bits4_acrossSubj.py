#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 15:40:45 2021

@author: ravinderjit
"""

import os
import pickle
import numpy as np
import scipy as sp
import mne
import matplotlib.pyplot as plt
from scipy.signal import freqz

data_loc = '/media/ravinderjit/Data_Drive/Data/EEGdata/TemporalCoding/AMmseq_bits4/'
pickle_loc = data_loc + 'Pickles/'


Subjects = ['S211','S207','S236','S228','S238'] #S237 data is crazy noisy

m_bits = [7,8,9,10]


fs = 4096

A_Tot_trials = []
A_Ht =[]
A_Htnf =[]

A_pca_sp = []
A_pca_coeff =[]
A_pca_expVar = []

A_pca_sp_nf = []
A_pca_coeff_nf = []
A_pca_expVar_nf = []

A_info_obj = []
A_ch_picks = []


#%% Load data

for sub in range(len(Subjects)):
    subject = Subjects[sub]
    with open(os.path.join(pickle_loc,subject+'_AMmseqbits4.pickle'),'rb') as file:
        [tdat, Tot_trials, Ht, Htnf, pca_sp, pca_coeff, pca_expVar, 
         pca_sp_nf, pca_coeff_nf,pca_expVar_nf,ica_sp,
         ica_coeff,ica_sp_nf,ica_coeff_nf, info_obj, ch_picks] = pickle.load(file)
    
    A_Tot_trials.append(Tot_trials)
    A_Ht.append(Ht)
    A_Htnf.append(Htnf)
    
    A_pca_sp.append(pca_sp)
    A_pca_coeff.append(pca_coeff)
    A_pca_expVar.append(pca_expVar)
    
    A_pca_sp_nf.append(pca_sp_nf)
    A_pca_coeff_nf.append(pca_coeff_nf)
    A_pca_expVar_nf.append(pca_expVar_nf)
    
    A_info_obj.append(info_obj)
    A_ch_picks.append(ch_picks)
    
    

#%% Check Sign of decomposition
for s in range(len(Subjects)):
    for m in range(len(m_bits)):
        if A_pca_coeff[s][m][0,-1] < 0:
            A_pca_coeff[s][m][0,:] = - A_pca_coeff[s][m][0,:]
            A_pca_sp[s][m][:,0] = - A_pca_sp[s][m][:,0]
        if A_pca_coeff[s][m][0,A_pca_coeff[s][m].shape[1]-2:].mean() < (A_pca_coeff[s][m][0,:].mean() - A_pca_coeff[s][m][0,:].std()):
            A_pca_coeff[s][m][0,:] = - A_pca_coeff[s][m][0,:]
            A_pca_sp[s][m][:,0] = - A_pca_sp[s][m][:,0]
            
            
for s in range(len(Subjects)):
    ch_ind = np.where(A_ch_picks[s] == 10)[0]
    for m in range(len(m_bits)):
        if A_pca_coeff[s][m][1,ch_ind] > 0:
            A_pca_coeff[s][m][1,:] = - A_pca_coeff[s][m][1,:]
            A_pca_sp[s][m][:,1] = - A_pca_sp[s][m][:,1]
        if A_pca_coeff[s][m][1,-1] < (A_pca_coeff[s][m][1,:].mean() - A_pca_coeff[s][m][1,:].std()):
            A_pca_coeff[s][m][1,:] = - A_pca_coeff[s][m][1,:]
            A_pca_sp[s][m][:,1] = - A_pca_sp[s][m][:,1]
                    

#%% Plot topomaps
    
plt.figure()
plt.suptitle('PCA comp 1')
vmin = A_pca_coeff[0][3][0,:].mean() - 1.96 *  A_pca_coeff[0][3][0,:].std()
vmax = A_pca_coeff[0][3][0,:].mean() + 1.96 *  A_pca_coeff[0][3][0,:].std()
for s in range(len(Subjects)):
    for m in range(len(m_bits)):
        plt.subplot(len(m_bits),len(Subjects),len(Subjects)*m+s+1)
        if m ==0:
            plt.title(Subjects[s])
        mne.viz.plot_topomap(A_pca_coeff[s][m][0,:], mne.pick_info(A_info_obj[s], A_ch_picks[s]),vmin=vmin,vmax=vmax)
        

plt.figure()
plt.suptitle('PCA comp 2')
vmin = A_pca_coeff[0][3][1,:].mean() - 1.96 *  A_pca_coeff[0][3][1,:].std()
vmax = A_pca_coeff[0][3][1,:].mean() + 1.96 *  A_pca_coeff[0][3][1,:].std()
for s in range(len(Subjects)):
    for m in range(len(m_bits)):
        plt.subplot(len(m_bits),len(Subjects),len(Subjects)*m+s+1)
        if m ==0:
            plt.title(Subjects[s])
        mne.viz.plot_topomap(A_pca_coeff[s][m][1,:], mne.pick_info(A_info_obj[s], A_ch_picks[s]),vmin=vmin,vmax=vmax)
        
#avg coeffs
pca_coeff_avg1 = np.zeros([32,len(m_bits)])
pca_coeff_avg2 = np.zeros([32,len(m_bits)])
chan_present = np.zeros([32])
for s in range(len(Subjects)):
    for m in range(len(m_bits)):
        pca_coeff_avg1[A_ch_picks[s],m] += A_pca_coeff[s][m][0,:]
        pca_coeff_avg2[A_ch_picks[s],m] += A_pca_coeff[s][m][1,:]
    chan_present[A_ch_picks[s]] += 1

pca_coeff_avg1 = pca_coeff_avg1 / chan_present[:,np.newaxis]
pca_coeff_avg2 = pca_coeff_avg2 / chan_present[:,np.newaxis]

plt.figure()
vmin = A_pca_coeff[0][3][0,:].mean() - 1.96 *  A_pca_coeff[0][3][0,:].std()
vmax = A_pca_coeff[0][3][0,:].mean() + 1.96 *  A_pca_coeff[0][3][0,:].std()
for m in range(len(m_bits)):
    plt.subplot(1,len(m_bits),m+1)
    mne.viz.plot_topomap(pca_coeff_avg1[:,m], mne.pick_info(A_info_obj[2], A_ch_picks[2]),vmin=vmin,vmax=vmax)
    plt.title(str(m_bits[m]) + 'bits')
plt.suptitle('Component 1')
        
plt.figure()
vmin = A_pca_coeff[0][3][0,:].mean() - 1.96 *  A_pca_coeff[0][3][0,:].std()
vmax = A_pca_coeff[0][3][0,:].mean() + 1.96 *  A_pca_coeff[0][3][0,:].std()
for m in range(len(m_bits)):
    plt.subplot(1,len(m_bits),m+1)
    mne.viz.plot_topomap(pca_coeff_avg2[:,m], mne.pick_info(A_info_obj[2], A_ch_picks[2]),vmin=vmin,vmax=vmax)
    plt.title(str(m_bits[m]) + 'bits')
plt.suptitle('Component 2')


#%% Extract NFs into numpy arrays
    
t = tdat[0]   
num_nf = len(A_Htnf[0]) / len(m_bits)
    
subj_pca1_nf = np.zeros([t.size,len(m_bits*num_nf),len(Subjects)])
subj_pca2_nf = np.zeros([t.size,len(m_bits*num_nf),len(Subjects)])
for s in range(len(Subjects)):
    for n in range(len(m_bits)*num_nf):
        subj_pca1_nf[:,n,s] = A_pca_sp_nf[s][n][:,0]
        subj_pca2_nf[:,n,s] = A_pca_sp_nf[s][n][:,1]


#%% Plot Time Domain bits
    
fig,ax = plt.subplots(4,len(Subjects),sharex=True)

for s in range(len(Subjects)):
    ax[0,s].set_title(Subjects[s])
    for m in range(4):
        ax[m,s].plot(t,A_pca_sp[s][m])
        ax[m,0].set_ylabel(str(m_bits[m]) + ' bits',rotation=0, size='large')
        
fig.suptitle('Ht pca both componenets ')   


fig,ax = plt.subplots(4,len(Subjects),sharex=True)
for s in range(len(Subjects)):
    ax[0,s].set_title(Subjects[s])
    for m in range(4):
        ax[m,s].plot(t,A_pca_sp[s][m][:,0])
        nf_mean = subj_pca1_nf[:,m*num_nf:(m+1)*num_nf,s].mean(axis=1)
        nf_std = subj_pca1_nf[:,m*num_nf:(m+1)*num_nf,s].std(axis=1)
        ax[m,s].plot(t,nf_mean,color='grey')
        ax[m,s].fill_between(t,nf_mean+1.96*nf_std,nf_mean-1.96*nf_std,color='grey',alpha=0.3)
    ax[0,s].set_title(Subjects[s])
    
fig.suptitle('Ht pca Comp 1')   
    
fig,ax = plt.subplots(4,len(Subjects),sharex=True)
for s in range(len(Subjects)):
    ax[m,0].set_ylabel(str(m_bits[m]) + ' bits',rotation=0, size='large')
    for m in range(4):
        ax[m,s].plot(t,A_pca_sp[s][m][:,1],color='tab:orange')
        nf_mean = subj_pca2_nf[:,m*num_nf:(m+1)*num_nf,s].mean(axis=1)
        nf_std = subj_pca2_nf[:,m*num_nf:(m+1)*num_nf,s].std(axis=1)
        ax[m,s].plot(t,nf_mean,color='grey')
        ax[m,s].fill_between(t,nf_mean+1.96*nf_std,nf_mean-1.96*nf_std,color='grey',alpha=0.3)
        ax[m,0].set_ylabel(str(m_bits[m]) + ' bits',rotation=0, size='large')
        
fig.suptitle('Ht pca Comp 2')   

pca_comp1_sp = np.zeros([t.size,len(m_bits),len(Subjects)])
pca_comp2_sp = np.zeros([t.size,len(m_bits),len(Subjects)])
for s in range(len(Subjects)):
    for m in range(4):
        pca_comp1_sp[:,m,s] = A_pca_sp[s][m][:,0]
        pca_comp2_sp[:,m,s] = A_pca_sp[s][m][:,1]
        
pca_comp1_sp_avg = pca_comp1_sp.mean(axis=2)
pca_comp2_sp_avg = pca_comp2_sp.mean(axis=2)
plt.figure()
for m in range(4):
    plt.subplot(1,len(m_bits),m+1)
    plt.plot(t,pca_comp1_sp_avg[:,m])
    # plt.plot(t,pca_comp2_sp_avg[:,m])
    
plt.figure()
for m in range(4):
    plt.subplot(1,len(m_bits),m+1)
    plt.plot(t,pca_comp2_sp_avg[:,m],color='tab:orange')
    
    
plt.figure()
plt.plot(t,pca_comp1_sp_avg[:,3])
plt.title('10 bits: Component 1')

plt.figure()
plt.plot(t,pca_comp2_sp_avg[:,3])
plt.title('10 bits: Component 2')

#%% Plot time domain Ht
    
fig,ax = plt.subplots(4,len(Subjects),sharex=True)
for s in range(len(Subjects)):
    for m in range(len(m_bits)):
        ax[m,s].plot(tdat[m],A_Ht[s][m][-1,:]) #Just Channel A32 (last channel)
    ax[0,s].set_title(Subjects[s])
    

sbp = [4,4]
sbp2 = [4,4]

for m in range(len(Ht)):
    fig,axs = plt.subplots(sbp[0],sbp[1],sharex=True,gridspec_kw=None)
    t = tdat[m]
    for s in range(len(Subjects)):
        Ht_1 = A_Ht[s][m]
        ch_picks_s = A_ch_picks[s]
        for p1 in range(sbp[0]):
            for p2 in range(sbp[1]):
                cur_ch = p1*sbp[1]+p2
                if np.any(cur_ch==ch_picks_s):
                    ch_ind = np.where(cur_ch==ch_picks_s)[0][0]
                    axs[p1,p2].plot(t,Ht_1[ch_ind,:])
                    axs[p1,p2].set_title(ch_picks[ch_ind])    
                    for n in range(int(m*num_nf),int(num_nf*(m+1))):
                        axs[p1,p2].plot(t,A_Htnf[s][n][ch_ind,:],color='grey',alpha=0.3)
                
    fig.suptitle('Ht ' + str(m_bits[m]) + ' bits')
    
for m in range(len(Ht)):    
    fig,axs = plt.subplots(sbp2[0],sbp2[1],sharex=True,gridspec_kw=None)
    t = tdat[m]
    for s in range(len(Subjects)):
        Ht_1 = A_Ht[s][m]
        ch_picks_s = A_ch_picks[s]
        for p1 in range(sbp2[0]):
            for p2 in range(sbp2[1]):
                cur_ch = p1*sbp2[1]+p2+sbp[0]*sbp[1]
                if np.any(cur_ch==ch_picks_s):
                    ch_ind = np.where(cur_ch==ch_picks_s)[0][0]
                    axs[p1,p2].plot(t,Ht_1[ch_ind,:])
                    axs[p1,p2].set_title(ch_picks[ch_ind])   
                    for n in range(int(m*num_nf),int(num_nf*(m+1))):
                        axs[p1,p2].plot(t,A_Htnf[s][n][ch_ind,:],color='grey',alpha=0.3)
            
    fig.suptitle('Ht ' +  str(m_bits[m]) + ' bits')   




#%% Plot Frequency Domian bits

t_end1 = int(np.round(.500 * fs))
t_end2 = int(np.round(.100*fs))
Nf = 8000
A_pca_Hf1 = np.empty([Nf,len(m_bits),len(Subjects)],dtype=complex)
A_pca_Hf2 = np.empty([Nf,len(m_bits),len(Subjects)],dtype=complex)
A_pca_phase1 = np.empty([Nf,len(m_bits),len(Subjects)])
A_pca_phase2 = np.empty([Nf,len(m_bits),len(Subjects)])

for s in range(len(Subjects)):
    for m in range(4):
        b_1 = A_pca_sp[s][m][:,0][:t_end1]
        b_2 = A_pca_sp[s][m][:,1][:t_end2]
        f,Hf_m1 = freqz(b_1,a=1,worN=Nf,fs=fs)
        f,Hf_m2 = freqz(b_2,a=1,worN=Nf,fs=fs)
        A_pca_Hf1[:,m,s] = Hf_m1
        A_pca_Hf2[:,m,s] = Hf_m2
        A_pca_phase1[:,m,s] = np.unwrap(np.angle(Hf_m1))
        A_pca_phase2[:,m,s] = np.unwrap(np.angle(Hf_m2))
        
        
fig,ax = plt.subplots(4,len(Subjects),sharex=True)
for s in range(len(Subjects)):
    ax[0,s].set_title(Subjects[s])
    for m in range(4):
        ax[m,s].plot(f,np.abs(A_pca_Hf1[:,m,s]))
        ax[m,s].plot(f,np.abs(A_pca_Hf2[:,m,s]))
        ax[m,s].set_xlim(0,100)
        
#from avg
b_1_avg = pca_comp1_sp_avg[:,3][:t_end1]
b_2_avg = pca_comp2_sp_avg[:,3][:t_end2]
f, Hf_avg1 = freqz(b_1_avg,a=1,worN=Nf,fs=fs)
f, Hf_avg2 = freqz(b_2_avg,a=1,worN=Nf,fs=fs)

plt.figure()
plt.plot(f,np.abs(Hf_avg1))
plt.plot(f,np.abs(Hf_avg2))
plt.xlim([0, 100])





#%% Calculate Group Delay
#Gonna compute in 10 Hz steps

#Plot phase response
fig,ax = plt.subplots(4,len(Subjects),sharex=True)
for s in range(len(Subjects)):
    ax[0,s].set_title(Subjects[s])
    for m in range(4):
        ax[m,s].plot(f,A_pca_phase1[:,m,s])
        ax[m,s].plot(f,A_pca_phase2[:,m,s])
        ax[m,s].set_xlim(0,100)
        ax[m,s].set_ylim(A_pca_phase1[:80,m,s].min(),np.max([A_pca_phase1[:80,m,s].max(),A_pca_phase2[:80,m,s].max()]))
        
phase_st = [4, 10, 20, 30, 40, 50, 60]
phase_ed = [10, 20, 30, 40, 50, 60, 70] 
        
A_GD_phase1 = np.zeros([len(phase_st),len(m_bits),len(Subjects)])
A_GD_phase2 = np.zeros([len(phase_st),len(m_bits),len(Subjects)])
for s in range(len(Subjects)):
    for m in range(4):
        for ph in range(len(phase_st)):
            f_ind1 = int(np.where(f>=phase_st[ph])[0][0])
            f_ind2 = int(np.where(f>=phase_ed[ph])[0][0])
            coeff1 = np.polyfit(f[f_ind1:f_ind2],A_pca_phase1[f_ind1:f_ind2,m,s],deg=1)
            coeff2 = np.polyfit(f[f_ind1:f_ind2],A_pca_phase2[f_ind1:f_ind2,m,s],deg=1)
            A_GD_phase1[ph,m,s] = -coeff1[0] / (2*np.pi)
            A_GD_phase2[ph,m,s] = -coeff2[0] / (2*np.pi)
            
#Plot Group Delays
fig,ax = plt.subplots(4,len(Subjects),sharex=True)
for s in range(len(Subjects)):
    ax[0,s].set_title(Subjects[s])
    for m in range(4):
        ax[m,s].plot(A_GD_phase1[:,m,s]*1000)
        ax[m,s].plot(A_GD_phase2[:,m,s]*1000)
        ax[m,s].set_xticks(range(len(phase_st)))
        ax[m,s].set_xticklabels(phase_ed)
fig.suptitle('Group Delay (ms)')



phase1_avg = np.unwrap(np.angle(Hf_avg1))
phase2_avg = np.unwrap(np.angle(Hf_avg2))

f_ind1 = int(np.where(f>=32)[0][0])
f_ind2 = int(np.where(f>=40)[0][0])

coeff1_avg = np.polyfit(f[f_ind1:f_ind2],phase1_avg[f_ind1:f_ind2],deg=1)
coeff2_avg = np.polyfit(f[f_ind1:f_ind2],phase2_avg[f_ind1:f_ind2],deg=1)

GD_1_avg = -coeff1_avg[0] / (2*np.pi)
GD_2_avg = -coeff2_avg[0] / (2*np.pi)
print(GD_1_avg*1000)
print(GD_2_avg*1000)

plt.figure()
plt.plot(f,phase1_avg)
plt.plot(f,phase2_avg)
plt.xlim([0, 100])
plt.ylim([-25,5])











    
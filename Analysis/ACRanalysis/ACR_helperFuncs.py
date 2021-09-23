#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 25 16:57:46 2021

@author: ravinderjit
"""

import numpy as np
from scipy.signal import freqz
from scipy.signal.windows import gaussian
from sklearn.decomposition import PCA
from sklearn.metrics import explained_variance_score
import matplotlib.pyplot as plt
import mne

def ACR_sourceHf(split_locs,ACR,t,fs,f1,f2):
    tpks = []
    pks = []
    pks_Hf = []
    pks_w = []
    pks_phase = []
    pks_phaseLine = []
    pks_phaseLineW = []
    pks_gd = []
    
    for pk in range(len(split_locs)):
        if pk ==0:
            t_1 = 0
        else:
            t_1 = split_locs[pk-1]
            
        t_2 = split_locs[pk]
        
        tpks.append(t[t_1:t_2])
        pks.append(ACR[t_1:t_2])
        
        [w, p_Hf] = freqz(b= ACR[t_1:t_2] - ACR[t_1:t_2].mean() ,a=1,worN=np.arange(0,fs/2,2),fs=fs)
        
        f_ind1 = np.where(w>=f1[pk])[0][0]
        f_ind2 = np.where(w>=f2[pk]+2)[0][0]
        
        phase_pkresp = np.unwrap(np.angle(p_Hf))
        coeffs= np.polyfit(w[f_ind1:f_ind2],phase_pkresp[f_ind1:f_ind2],deg=1)
        pks_phaseLine.append(coeffs[0] * w[f_ind1:f_ind2] +coeffs[1])
        pks_phaseLineW.append(w[f_ind1:f_ind2])
        pks_gd.append(-coeffs[0] / (2*np.pi))
        
        pks_w.append(w)
        pks_Hf.append(p_Hf)
        pks_phase.append(phase_pkresp)
    
        
    return tpks, pks, pks_Hf, pks_w, pks_phase, pks_phaseLine, pks_phaseLineW, pks_gd
    

def ACR_model(latency,width, weights,latency_pad,fs):
    full_mod = np.array([])
    stds = width / 4 #SD of the sources in seconds
    
    source_mods = []
    for source in range(stds.size):
        gauss_source = gaussian(np.round(width[source]*fs),np.round(stds[source]*fs))
        source_mods.append(gauss_source - np.min(gauss_source))
        
    
    
    for source in range(len(source_mods)):
    
        if source == 2:
            continue
        
        if (source == 1): # do mixed source
    
            s2 = np.concatenate((np.zeros( int(np.round( (np.sum(width[:2]) + width[2]/2 - latency[2]) * fs)) ), source_mods[2]  ))
            s1 = np.concatenate((source_mods[1], np.zeros(s2.size-source_mods[1].size)))
            mixed_source = weights[1]*s1 + weights[2]*s2
            
            full_mod = np.append(full_mod, mixed_source)
            
        else:
            full_mod = np.append(full_mod,np.min(source_mods[source])*np.ones(int(np.round(latency_pad[source]*fs))))
            full_mod = np.append(full_mod, weights[source] * source_mods[source])


    return full_mod



def PCA_tcuts(Ht,t,t_cuts,ch_picks,chs_use):
    pca = PCA(n_components=2)
    pca_sp_cuts_ = []
    pca_expVar_cuts_ = []
    pca_coeff_cuts_ = []
    t_cuts_ = []
    for t_c in range(len(t_cuts)):
        if t_c ==0:
            t_1 = np.where(t>=0)[0][0]
        else:
            t_1 = np.where(t>=t_cuts[t_c-1])[0][0]
        
        t_2 = np.where(t>=t_cuts[t_c])[0][0]
        
        pca_sp = pca.fit_transform(Ht[chs_use,t_1:t_2].T)
        pca_expVar = pca.explained_variance_ratio_
        pca_coeff = pca.components_
        
        if pca_coeff[0,ch_picks[chs_use]==31] < 0:  #Consider to Expand this too look at mutlitple electrodes
           pca_coeff = -pca_coeff
           pca_sp = -pca_sp
        
        pca_sp_cuts_.append(pca_sp)
        pca_expVar_cuts_.append(pca_expVar)
        pca_coeff_cuts_.append(pca_coeff)
        t_cuts_.append(t[t_1:t_2])
        
    return pca_sp_cuts_, pca_expVar_cuts_, pca_coeff_cuts_, t_cuts_
    

def Template_tcuts(Ht, t, t_cuts, ch_picks, template):
    t_tc = []
    template_tc_sp = []
    template_tc_expVar = []
    for t_c in range(len(t_cuts)):
        if t_c ==0:
            t_1 = np.where(t>=0)[0][0]
        else:
            t_1 = np.where(t>=t_cuts[t_c-1])[0][0]
        
        t_2 = np.where(t>=t_cuts[t_c])[0][0]
        
        resp = Ht[:,t_1:t_2] - Ht[:,t_1:t_2].mean(axis=1)[:,np.newaxis]
        
        template_sp = np.matmul(resp.T, template[ch_picks])

        template_est = np.matmul(template[ch_picks,np.newaxis],template_sp[np.newaxis,:])

        template_expVar = explained_variance_score(resp, template_est, multioutput='variance_weighted')  

        t_tc.append(t[t_1:t_2])
        
        template_tc_sp.append(template_sp)
        template_tc_expVar.append(template_expVar)
    
    return t_tc, template_tc_sp, template_tc_expVar


def PCA_tcuts_topomap(pca_coeffCuts, t_cuts, pca_expVarCuts, infoObj, ch_picks, title_):
    plt.figure()
    plt.suptitle(title_)
    vmin = pca_coeffCuts[-1][0,:].mean() - 2 * pca_coeffCuts[-1][0,:].std()
    vmax = pca_coeffCuts[-1][0,:].mean() + 2 * pca_coeffCuts[-1][0,:].std()
    for t_c in range(len(t_cuts)):
        plt.subplot(2,len(t_cuts),t_c+1)
        plt.title('ExpVar ' + str(np.round(pca_expVarCuts[t_c][0]*100)) + '%')
        mne.viz.plot_topomap(pca_coeffCuts[t_c][0,:], mne.pick_info(infoObj,ch_picks),vmin=vmin,vmax=vmax)
        
        plt.subplot(2,len(t_cuts),t_c+1 + len(t_cuts))
        plt.title('ExpVar ' + str(np.round(pca_expVarCuts[t_c][1]*100)) + '%')
        mne.viz.plot_topomap(pca_coeffCuts[t_c][1,:], mne.pick_info(infoObj,ch_picks),vmin=vmin,vmax=vmax)













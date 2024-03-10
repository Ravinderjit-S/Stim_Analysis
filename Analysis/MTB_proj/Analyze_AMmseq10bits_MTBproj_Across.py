#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 14:56:46 2023

@author: ravinderjit
"""


import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import mne
from anlffr.preproc import find_blinks
from EEGpp import EEGconcatenateFolder
from mne.preprocessing.ssp import compute_proj_epochs
import os
import pickle
import sys
sys.path.append(os.path.abspath('../mseqAnalysis/'))
from mseqHelper import mseqXcorr
from mseqHelper import mseqXcorrEpochs_fft
from sklearn.decomposition import PCA
from itertools import compress
from scipy.signal import find_peaks
from scipy.stats import pearsonr


#%% Functions

def central_gain_modTRF(t_epochs, modTRF_150):
    #take in modTRF_150 and baseline to time zero and normalize amplitude to first peak
    #Signal should be time x channels 
    
    t_0 = np.where(t_epochs>=0)[0][0]
    t_15 = np.where(t_epochs>=0.01)[0][0] #First peak occurs in first 15 ms
    
    cg_modTRF = np.empty(modTRF_150.shape)
    
    for ch in range(modTRF_150.shape[1]):
        this_ch = modTRF_150[:,ch] - modTRF_150[t_0,ch] #baseline to time 0
        max_loc = np.argmax(np.abs(this_ch[t_0:t_15])) + t_0
        this_ch = this_ch / np.abs(this_ch[max_loc]) #normalize to first peak
        
        cg_modTRF[:,ch] = this_ch
    
    return cg_modTRF
        
    

def plot_age(t_epochs, signal, age, title_):
    #Generate two figures showing signal changing across age
    #Signal is time x subjects
    
    age_1 = age <25 
    age_2 = np.logical_and(age >=25, age < 36)
    age_3 = np.logical_and(age >35, age <56)
    age_4 = age >=56

    mean_1 = signal[:,age_1].mean(axis=1)
    mean_2 = signal[:,age_2].mean(axis=1)
    mean_3 = signal[:,age_3].mean(axis=1)
    mean_4 = signal[:,age_4].mean(axis=1)

    std_err1 = signal[:,age_1].std(axis=1) / np.sqrt(np.sum(age_1))
    std_err2 = signal[:,age_2].std(axis=1) / np.sqrt(np.sum(age_2))
    std_err3 = signal[:,age_3].std(axis=1) / np.sqrt(np.sum(age_3))
    std_err4 = signal[:,age_4].std(axis=1) / np.sqrt(np.sum(age_4))

    plt.figure()
    plt.plot(t_epochs, mean_1,label='< 25', color='b')
    plt.fill_between(t_epochs, mean_1 - std_err1, mean_1 + std_err1, alpha=0.5, color='b')

    plt.plot(t_epochs, mean_2,label='25-35', color='g')
    plt.fill_between(t_epochs, mean_2 - std_err2, mean_2 + std_err2, alpha=0.5, color='g')

    plt.plot(t_epochs, mean_3,label='35-55', color='orange')
    plt.fill_between(t_epochs, mean_3 - std_err3, mean_3 + std_err3, alpha=0.5, color='orange')

    plt.plot(t_epochs, mean_4,label='>56', color='r')
    plt.fill_between(t_epochs, mean_4 - std_err4, mean_4 + std_err4, alpha=0.5, color='r')

    plt.xlim([-.05,0.5])
    plt.xlabel('Time (sec)')
    plt.title(title_)
    plt.legend()
    
    # plt.figure()
    # plt.plot(t_epochs, signal[:,age_1], color='b')
    # plt.plot(t_epochs, signal[:,age_2], color='g')
    # plt.plot(t_epochs, signal[:,age_3], color='orange')
    # plt.plot(t_epochs, signal[:,age_4], color='r')
    # plt.xlim([-.05,0.5])
    # plt.xlabel('Time (sec)')
    # plt.title(title_)
    
    
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


def plot_quartiles(t_epochs, signal, metric, title_):
    #Signal is time x subjects
    
    q1 = metric <= np.percentile(metric,25)
    q2 = np.logical_and(metric > np.percentile(metric,25),  metric <= np.percentile(metric, 50))
    q3 = np.logical_and(metric > np.percentile(metric,50),  metric <= np.percentile(metric, 75))
    q4 = metric > np.percentile(metric,75)
    
    mean_1 = signal[:,q1].mean(axis=1)
    mean_2 = signal[:,q2].mean(axis=1)
    mean_3 = signal[:,q3].mean(axis=1)
    mean_4 = signal[:,q4].mean(axis=1)

    std_err1 = signal[:,q1].std(axis=1) / np.sqrt(np.sum(q1))
    std_err2 = signal[:,q2].std(axis=1) / np.sqrt(np.sum(q2))
    std_err3 = signal[:,q3].std(axis=1) / np.sqrt(np.sum(q3))
    std_err4 = signal[:,q4].std(axis=1) / np.sqrt(np.sum(q4))

    plt.figure()
    plt.plot(t_epochs, mean_1,label='q1', color='b')
    plt.fill_between(t_epochs, mean_1 - std_err1, mean_1 + std_err1, alpha=0.5, color='b')

    plt.plot(t_epochs, mean_2,label='q2', color='g')
    plt.fill_between(t_epochs, mean_2 - std_err2, mean_2 + std_err2, alpha=0.5, color='g')

    plt.plot(t_epochs, mean_3,label='q3', color='orange')
    plt.fill_between(t_epochs, mean_3 - std_err3, mean_3 + std_err3, alpha=0.5, color='orange')

    plt.plot(t_epochs, mean_4,label='q4', color='r')
    plt.fill_between(t_epochs, mean_4 - std_err4, mean_4 + std_err4, alpha=0.5, color='r')

    plt.xlim([-.05,0.5])
    plt.xlabel('Time (sec)')
    plt.title(title_)
    plt.legend()
    
def AvgCgChannels(t_epochs, A_sig, A_chPicks, Num_chs):
    #Average signal for each channel across subjects after normalizing to compute central gain.
    #Some Participants will be missing some channels so will account for that
    #A_sig is list of participants with each item being channls x time (... modify this function to allow each item to be time x channels)
    
    Ht_avg = np.zeros((t_epochs.size, Num_chs ))
    ch_count = np.zeros(Num_chs)
    
    for s in range(len(A_sig)):
        Ht = central_gain_modTRF(t_epochs, A_sig[s].T)
        chPicks = A_chPicks[s]  
        for ss in range(chPicks.size):
            Ht_avg[:,chPicks[ss]] += Ht[:,ss] 
            ch_count[chPicks[ss]] +=1
        
    Ht_avg = Ht_avg / ch_count[:,np.newaxis].T
    
    return Ht_avg

def plot32Chans(t_epochs, sig, ch_picks, title_):
    #sig is time x channels 
    
    fig, ax = plt.subplots(4,4,sharex=True)
    for ii in ch_picks[ch_picks<16]:
        if ii >15:
            break
        ind1 = np.mod(ii,4)
        ind2 = int(np.floor(ii/4))
        ax[ind2,ind1].plot(t_epochs,sig[:,ii])
        ax[ind2,ind1].set_title(ii)
        ax[ind2,ind1].set_xlim(-0.050, 0.5)
    fig.suptitle(title_)
            
        
    fig, ax = plt.subplots(4,4,sharex=True)
    for ii in ch_picks[ch_picks>=16]:
        ind1 = np.mod(ii,4)
        ind2 = int(np.floor(ii/4)) - 4
        ax[ind2,ind1].plot(t_epochs,sig[:,ii])
        ax[ind2,ind1].set_title(ii)
        ax[ind2,ind1].set_xlim(-0.050, 0.5)
    fig.suptitle(title_)
    
def peakAnalysis(t_epochs, sig, ch_picks, plotit):
    #Sig is time x channels 
    #Find location of the 5 expected peaks in mod-TRF
    #plotit = boolean
    
    pk_indexes = np.zeros(5,dtype=(int))
    pk_latencys = np.zeros(5,dtype=(float))
    pkn_indexes= np.zeros(5,dtype=(int))
    pk_widths = np.zeros(5,dtype=(float))
    
    pk_post_index = np.zeros(1,dtype=(int))
    pk_post_latency = np.zeros(1,dtype=(float))
    pkn_post_indexes = np.zeros(2,dtype=(int))
    
    chans_cent = [4,25,31,8,21]
    chans_post = [14,15,16,13,17]
    
    sig_cent = sig[:,np.in1d(ch_picks,chans_cent)].mean(axis=1)
    sig_post = sig[:,np.in1d(ch_picks,chans_post)].mean(axis=1)
    
    t_0 = np.where(t_epochs>=0)[0][0]
    fs = 4096
    peaks, prop = find_peaks(sig_cent[t_0:],prominence=(None,None),width=(None,None))
    peaks_neg, x = find_peaks(-sig_cent[t_0:])
    
    peaks_post, x = find_peaks(sig_post[t_0:])
    peaks_neg_post, x = find_peaks(-sig_post[t_0:])
    
    pks = peaks[0:6] + t_0
    pksn = peaks_neg[0:6] + t_0
    
    ppks = peaks_post[0:6] + t_0
    ppksn = peaks_neg_post[0:6] + t_0
    
    #First peak
    pk_indexes[0] = peaks[0] + t_0
    pk_latencys[0] = peaks[0] / fs
    pkn_indexes[0] = peaks_neg[0] + t_0
    pk_widths[0] = peaks_neg[0] /fs
    
    #Peak 2a -- in posterior
    pk_post_index = peaks_post[1] + t_0
    pk_post_latency = peaks_post[1] / fs
    pkn_post_indexes = peaks_neg_post[0:2] + t_0
    
    #Peak 2b & Peak 3
    # These peaks can be difficult to separate in some instances so additional analysis to separate them

    if ((pksn[1] - pksn[0]) /fs > 0.035):
        # Use 2nd derivitive to separate the two peaks
        der2 = np.diff(np.diff(sig_cent))
        
        #After the first negative pk, where does the second derivitive say the signal is concave down
        con_down_1 = np.where(der2[pksn[0]:]<0)[0][0] + pksn[0] 
        
        #After that concave down point, where does the signal become concave up again. This will be the point used to split the two sources
        con_up_1 = np.where(der2[con_down_1:] >0)[0][0] + con_down_1 

        
        pk_indexes[1] = con_up_1 + 2 #+2 since point is from 2nd derivitive. negligible to account for
        pk_latencys[1] = (pk_indexes[1] - t_0) / fs
        pkn_indexes[1] = pk_indexes[1]
        pk_widths[1] = (pk_indexes[1] - t_0 - peaks_neg[0]) / fs 

        peaks = np.insert(peaks,1,pk_indexes[1]-t_0)
        peaks_neg = np.insert(peaks_neg,1,pk_indexes[1]-t_0)

        pk_indexes[2] = peaks[2] + t_0
        pk_latencys[2] = peaks[2] / fs
        pkn_indexes[2] = peaks_neg[2] + t_0
        pk_widths[2] = (peaks_neg[2] - peaks_neg[1]) / fs 

    else:
        pk_indexes[1] = peaks[1] + t_0
        pk_latencys[1] = peaks[1] / fs
        pkn_indexes[1] = peaks_neg[1] + t_0
        pk_widths[1] = (peaks_neg[1] - peaks_neg[0]) / fs 
    
        pk_indexes[2] = peaks[2] + t_0
        pk_latencys[2] = peaks[2] / fs
        pkn_indexes[2] = peaks_neg[2] + t_0
        pk_widths[2] = (peaks_neg[2] - peaks_neg[1]) / fs 
        
    #Peak 4
    pk_indexes[3] = peaks[3] + t_0
    pk_latencys[3] = peaks[3] /fs
    pkn_indexes[3] = peaks_neg[3] + t_0
    pk_widths[3] = (peaks_neg[3] - peaks_neg[2]) / fs
    
    #Peak 5
    #convolving with rectangle to smooth out peak detection
    rec = np.ones(int(np.round(fs*0.020)))
    sig_conv = np.convolve(sig_cent,rec,mode='same') / np.sum(rec)
    
    peaks_conv,x = find_peaks(sig_conv[pk_indexes[3]:])
    peaks_neg_conv,x = find_peaks(-sig_conv[pk_indexes[3]:])
        
    pk_indexes[4] = peaks_conv[0] + pk_indexes[3]
    pk_latencys[4] = (peaks_conv[0] + pk_indexes[3] - t_0) / fs
    pkn_indexes[4] = peaks_neg_conv[1] + pk_indexes[3]
    pk_widths[4] = (peaks_neg_conv[1] + pk_indexes[3] - t_0 - peaks_neg[3]) /fs
    
    if plotit:
        plt.figure()
        plt.plot(t_epochs,sig_cent)
        #plt.plot(t_epochs,sig_conv)
        plt.plot(t_epochs[pk_indexes], sig_cent[pk_indexes], 'x', color='r')
        plt.plot(t_epochs[pkn_indexes], sig_cent[pkn_indexes], 'x', color='b')
        plt.plot(t_epochs[pk_post_index], sig_cent[pk_post_index], 'x', color='purple')
        plt.xlim((-0.05,0.5))
        
        plt.figure()
        plt.plot(t_epochs, sig_post)
        plt.plot(t_epochs[pk_indexes], sig_post[pk_indexes], 'x', color='r')
        plt.plot(t_epochs[pkn_indexes], sig_post[pkn_indexes], 'x', color='b')
        plt.plot(t_epochs[pk_post_index], sig_post[pk_post_index], 'x', color='purple')
        plt.plot(t_epochs[pkn_post_indexes], sig_post[pkn_post_indexes], 'x', color='pink')
        plt.xlim((-0.05,0.5))
    
 
    return pk_indexes, pkn_indexes, pk_latencys, pk_widths, pk_post_index, pk_post_latency, pkn_post_indexes
    
    
def modTRFSources(t_epochs, sig, i_cuts, ch_picks, plotit, info_obj):
    #sig is time x channels
    #i_cuts: indexes for where to cut out sources
    #i_cuts_post: indexes for one source in the posterior -- took out for now
    #plotit: boolean
    
    pca_tcuts = []
    pca_coeffs = []
    pca_expVar = []
    times = []
    
    for i_c in range(1,len(i_cuts)):
        pca = PCA(n_components=1)
        
        i_1 = i_cuts[i_c-1]
        i_2 = i_cuts[i_c]
        
        source = pca.fit_transform(sig[i_1:i_2,:])
        expVar = pca.explained_variance_ratio_
        coeff = pca.components_
        
        if (coeff[0,ch_picks==31] < 0):
            coeff = -coeff
            source = -source
        
        pca_tcuts.append(source)
        pca_coeffs.append(coeff)
        pca_expVar.append(expVar)
        times.append(t_epochs[i_1:i_2])
        
    # Posterior source
    # pca = PCA(n_components=1)
    
    # i_1 = i_cuts_post[0]
    # i_2 = i_cuts_post[1]
    
    # source = pca.fit_transform(sig[i_1:i_2,:])
    # expVar = pca.explained_variance_ratio_
    # coeff = pca.components_
    
    # if (coeff[0,ch_picks==15] < 0):
    #     coeff = -coeff
    #     source = -source
    
    # times_post = t_epochs[i_1:i_2]
    # pca_post = source
    # pca_post_coeffs = coeff
    # pca_post_expVar = expVar
    
    if plotit:
        plt.figure()
        for t_c in range(len(times)):
            plt.plot(times[t_c], pca_tcuts[t_c])
        
        #plt.plot(times_post,pca_post, color='k')
        
        fig = plt.figure()
        fig.set_size_inches(9,4)
        vmin = pca_coeffs[-1][0,:].mean() - 2 * pca_coeffs[-1][0,:].std()
        vmax = pca_coeffs[-1][0,:].mean() + 2 * pca_coeffs[-1][0,:].std()
        for t_c in range(len(times)):
            ax = plt.subplot(1,len(times),t_c+1)
            plt.title('ExpVar ' + str(np.round(pca_expVar[t_c][0]*100)) + '%')
            mne.viz.plot_topomap(pca_coeffs[t_c][0,:], mne.pick_info(info_obj,ch_picks),vlim=(vmin,vmax),axes=ax)
        #posterior source
        # ax = plt.subplot(1,len(times)+1,len(times)+1)
        # plt.title('Post ExpVar ' + str(np.round(pca_post_expVar*100)) + '%')
        # mne.viz.plot_topomap(pca_post_coeffs[0,:], mne.pick_info(info_obj,ch_picks),vlim=(vmin,vmax),axes=ax)
    
    return times, pca_tcuts, pca_coeffs, pca_expVar, #times_post #pca_post, pca_post_coeffs, pca_post_expVar
        
        
def TwoPeaks(t_epochs, mod_trf):
    #mod_trf is 1d 
    
    t_10 = np.where(t_epochs>.010)[0][0]
    t_70 = np.where(t_epochs>.070)[0][0]
    t_400 = np.where(t_epochs>0.4)[0][0]
    
    peaks, x = find_peaks(mod_trf[t_10:t_70])
    max_x = np.argmax(mod_trf[peaks+t_10])
    max1 = peaks[max_x] + t_10
    
    peaks, x = find_peaks(mod_trf[t_70:t_400])
    max_x = np.argmax(mod_trf[peaks+t_70])
    max2 = peaks[max_x] + t_70
    
    maxlocs = np.array([max1, max2])
    
    return maxlocs
    
    



#%% Data

data_loc = '/media/ravinderjit/Data_Drive/Data/EEGdata/MTB/mTRF/'
pickle_loc = data_loc + 'Pickles/'

fig_loc =  '/media/ravinderjit/Data_Drive/Data/Figures/MTBproj/mTRF/'

Subjects = ['S069', 'S072', 'S078', 'S088', 'S104', 'S105', 'S259', 'S260', 'S268', 'S269',
            'S270', 'S271', 'S273', 'S274', 'S277', 'S279', 'S280', 'S281', 'S282', 'S284', 
            'S285', 'S288', 'S290', 'S291', 'S303', 'S305', 'S308', 'S310', 'S312', 'S337', 
            'S339', 'S340', 'S341', 'S342', 'S344', 'S345', 'S347', 'S352', 'S355', 'S358']

age = np.array([49, 55, 47, 52, 51, 61, 20, 33, 19, 19, 
       21, 21, 18, 19, 20, 20, 20, 21, 19, 26,
       19, 30, 21, 66, 28, 27, 59, 70, 37, 66,
       71, 39, 35, 54, 60, 61, 38, 35, 49, 56 ])

chs_model = np.arange(32)

Acz_evk = np.empty([8192,len(Subjects),])
A_Hts = []
A_chPicks = []
A_refchans = []
A_numEpochs = []

#%% Load data

for s in range(len(Subjects)):
    subject = Subjects[s]
    with open(os.path.join(pickle_loc,subject +'_AMmseq10bits_epochs.pickle'),'rb') as file:
        [Ht_epochs,t_epochs, refchans, ch_picks, info_obj] = pickle.load(file)
        Cz_evkd = Ht_epochs[-1,:,:].mean(axis=0)
        Cz_evkd = Cz_evkd[:,np.newaxis]
        
    # t_0 = np.where(t_epochs>=0)[0][0]
    # t_15 = np.where(t_epochs>=0.015)[0][0]
    # Cz_evkd = Cz_evkd - Cz_evkd[t_0] #baseline to time 0
    # Cz_evkd = Cz_evkd / np.max(Cz_evkd[t_0:t_15]) #normalize to first peak
    
    Cz_evkd = central_gain_modTRF(t_epochs, Cz_evkd)
    
    Acz_evk[:,s] = Cz_evkd[:,0]
    A_numEpochs.append(Ht_epochs.shape[1])
    A_Hts.append(Ht_epochs.mean(axis=1))
    A_chPicks.append(ch_picks)
    A_refchans.append(refchans)
    
#%% Plot all responses        
plt.figure()
plt.plot(t_epochs,Acz_evk)
plt.plot(t_epochs,Acz_evk.mean(axis=1),color='k',linewidth=2)
plt.xlim([-0.05,0.5])

#%% Plot responses by age

plot_age(t_epochs, Acz_evk, age, 'CZevkd')


#%% Look at 5 Regions of Scalp

chans_frnt = [0,1,28,29]
chans_cent = [4,25,31,8,21]
chans_post = [14,15,16,13,17]
chans_tl = [6,5,9]
chans_tr = [23, 24, 20]

A_frnt = np.empty([8192,len(Subjects),])
A_center = np.empty([8192,len(Subjects),])
A_post = np.empty([8192,len(Subjects),])
A_tl = np.empty([8192,len(Subjects),])
A_tr = np.empty([8192,len(Subjects),])

for s in range(len(Subjects)):
    ch_picks = A_chPicks[s]
    Ht = A_Hts[s]
    
    Ht = central_gain_modTRF(t_epochs, Ht.T)
    
    frnt_mask = np.isin(ch_picks, chans_frnt)
    cent_mask = np.isin(ch_picks, chans_cent)
    post_mask = np.isin(ch_picks, chans_post)
    tl_mask = np.isin(ch_picks, chans_tl)
    tr_mask = np.isin(ch_picks, chans_tr)
    
    avg_frnt = Ht[:,frnt_mask].mean(axis=1)
    avg_center = Ht[:,cent_mask].mean(axis=1)
    avg_post = Ht[:,post_mask].mean(axis=1)
    avg_tl = Ht[:,tl_mask].mean(axis=1)
    avg_tr = Ht[:,tr_mask].mean(axis=1)
    
    A_frnt[:,s] = avg_frnt
    A_center[:,s] = avg_center
    A_post[:,s] = avg_post
    A_tl[:,s] = avg_tl
    A_tr[:,s] = avg_tr
    

plt.figure()
plt.plot(t_epochs,A_frnt)

plt.figure()
plt.plot(t_epochs, A_center)

plt.figure()
plt.plot(t_epochs,A_center.mean(axis=1),color='tab:blue', label='Center')
plt.plot(t_epochs,-A_frnt.mean(axis=1),color='tab:green', label='Front')
plt.plot(t_epochs,A_post.mean(axis=1),color='tab:purple', label = 'Post')
plt.plot(t_epochs,A_tl.mean(axis=1),color='tab:orange', label = 'TL')
plt.plot(t_epochs,A_tr.mean(axis=1),color='tab:red', label = 'TR')
plt.xlim([-0.05,0.5])
plt.legend()

plot_age(t_epochs,A_frnt,age,'Frontal')
plot_age(t_epochs,A_center,age,'Central')
plot_age(t_epochs,A_post,age,'Posterior')
plot_age(t_epochs,A_tl,age,'TL')
plot_age(t_epochs,A_tr,age,'TR')

# plt.figure()
# plt.plot(t_epochs,avg_center,color='tab:blue')
# plt.plot(t_epochs,-avg_frnt, color='tab:green')
# plt.plot(t_epochs,avg_post, color='tab:purple')
# plt.plot(t_epochs,avg_tl, color='tab:orange')
# plt.plot(t_epochs,avg_tr, color='tab:red')
# plt.xlim([-0.05,0.5])


#%% SIN and mod-TRF

data_loc_JANE = '/media/ravinderjit/Data_Drive/Data/MTB_Behavior/SIN_Info_JANE/SINinfo_Jane.mat'
jane = sio.loadmat(data_loc_JANE,squeeze_me = True)

jane_subs = list(jane['Subjects'])
jane_thresh = jane['thresholds']

index_sub_jane, del_inds_jane = SortSubjects(Subjects, jane_subs)
jane_age = age[index_sub_jane]
jane_thresh = np.delete(jane_thresh,del_inds_jane)

Acz_evk_jane = Acz_evk[:,index_sub_jane]
plot_quartiles(t_epochs, Acz_evk_jane, jane_thresh, 'Jane')



data_loc_MRT = '/media/ravinderjit/Data_Drive/Stim_Analysis/Analysis/SnapLabOnline/MTB_MRT_online/MTB_MRT.mat'
mrt = sio.loadmat(data_loc_MRT,squeeze_me = True)

mrt_subs = list(mrt['Subjects'])
mrt_thresh = mrt['thresholds']

index_sub_mrt, del_inds_mrt = SortSubjects(Subjects, mrt_subs)
mrt_age = age[index_sub_mrt]
mrt_thresh = np.delete(mrt_thresh,del_inds_mrt)

Acz_evk_mrt = Acz_evk[:,index_sub_mrt]
plot_quartiles(t_epochs, Acz_evk_mrt, mrt_thresh, 'MRT')



#%% modTRF source analysis by age

age_1 = age <25 
age_2 = np.logical_and(age >=25, age < 36)
age_3 = np.logical_and(age >35, age <56)
age_4 = age >=56

Ht_avg_1 = AvgCgChannels(t_epochs, list(compress(A_Hts, age_1)), list(compress(A_chPicks,age_1)), 32)
Ht_avg_2 = AvgCgChannels(t_epochs, list(compress(A_Hts, age_2)), list(compress(A_chPicks,age_2)), 32)
Ht_avg_3 = AvgCgChannels(t_epochs, list(compress(A_Hts, age_3)), list(compress(A_chPicks,age_3)), 32)
Ht_avg_4 = AvgCgChannels(t_epochs, list(compress(A_Hts, age_4)), list(compress(A_chPicks,age_4)), 32)

plot32Chans(t_epochs, Ht_avg_1, ch_picks, 'youngest')
plot32Chans(t_epochs, Ht_avg_2, ch_picks, 'not youngest')

fig, ax = plt.subplots(4,4,sharex=True)
ch_picks = np.arange(32)
for ii in ch_picks[ch_picks<16]:
    if ii >15:
        break
    ind1 = np.mod(ii,4)
    ind2 = int(np.floor(ii/4))
    ax[ind2,ind1].plot(t_epochs,Ht_avg_1[:,ii], color='b')
    ax[ind2,ind1].plot(t_epochs,Ht_avg_2[:,ii], color='g')
    ax[ind2,ind1].plot(t_epochs,Ht_avg_3[:,ii], color='orange')
    ax[ind2,ind1].plot(t_epochs,Ht_avg_4[:,ii], color='red')

    ax[ind2,ind1].set_title(ii)
    ax[ind2,ind1].set_xlim(-0.050, 0.5)

    
fig, ax = plt.subplots(4,4,sharex=True)
for ii in ch_picks[ch_picks>=16]:
    ind1 = np.mod(ii,4)
    ind2 = int(np.floor(ii/4)) - 4
    ax[ind2,ind1].plot(t_epochs,Ht_avg_1[:,ii], color='b')
    ax[ind2,ind1].plot(t_epochs,Ht_avg_2[:,ii], color='g')
    ax[ind2,ind1].plot(t_epochs,Ht_avg_3[:,ii], color='orange')
    ax[ind2,ind1].plot(t_epochs,Ht_avg_4[:,ii], color='red')
    
    ax[ind2,ind1].set_title(ii)
    ax[ind2,ind1].set_xlim(-0.050, 0.5)


chans_cent = [4,25,31,8,21]
cent_avg_1 = Ht_avg_1[:,chans_cent].mean(axis=1)
cent_avg_2 = Ht_avg_2[:,chans_cent].mean(axis=1)

t_0 = np.where(t_epochs >=0)[0][0]

pkag_indexes = np.zeros((5,4),dtype=int)
pknag_indexes = np.zeros((5,4),dtype=int)
pkag_latencys = np.zeros((5,4))
pkag_widths = np.zeros((5,4))
pkag_amp_cz = np.zeros((5,4))

[pkag_indexes[:,0], pknag_indexes[:,0], pkag_latencys[:,0], pkag_widths[:,0], pk_post_index, pk_post_latency, pkn_post_indexes] = peakAnalysis(t_epochs, Ht_avg_1, np.arange(32), False)
pkag_amp_cz[:,0] = Ht_avg_1[pkag_indexes[:,0],31]
[pkag_indexes[:,1], pknag_indexes[:,1], pkag_latencys[:,1], pkag_widths[:,1], pk_post_index, pk_post_latency, pkn_post_indexes] = peakAnalysis(t_epochs, Ht_avg_2, np.arange(32), False)    
pkag_amp_cz[:,1] = Ht_avg_2[pkag_indexes[:,1],31]
[pkag_indexes[:,2], pknag_indexes[:,2], pkag_latencys[:,2], pkag_widths[:,2], pk_post_index, pk_post_latency, pkn_post_indexes] = peakAnalysis(t_epochs, Ht_avg_3, np.arange(32), False)
pkag_amp_cz[:,2] = Ht_avg_3[pkag_indexes[:,2],31]
[pkag_indexes[:,3], pknag_indexes[:,3], pkag_latencys[:,3], pkag_widths[:,3], pk_post_index, pk_post_latency, pkn_post_indexes] = peakAnalysis(t_epochs, Ht_avg_4, np.arange(32), False)
pkag_amp_cz[:,3] = Ht_avg_4[pkag_indexes[:,3],31]

plt.figure()
plt.plot(np.arange(1,6),pkag_latencys, 'x')
plt.legend(['Young', 'Less young', 'Middle Age', 'Old'])

pknag_latencys = t_epochs[pknag_indexes]
plt.figure()
plt.plot(np.arange(1,6), pknag_latencys, 'x')
plt.legend(['Young', 'Less young', 'Middle Age', 'Old'])

plt.figure()
plt.plot(np.arange(1,6),pkag_amp_cz)


[times, pca_tcuts, pca_coeffs, pca_expVar] = modTRFSources(t_epochs, Ht_avg_1, np.insert(pknag_indexes[:,0], 0, t_0), ch_picks, True, info_obj)
[times, pca_tcuts, pca_coeffs, pca_expVar] = modTRFSources(t_epochs, Ht_avg_2, np.insert(pknag_indexes[:,1], 0, t_0), ch_picks, True, info_obj)
[times, pca_tcuts, pca_coeffs, pca_expVar] = modTRFSources(t_epochs, Ht_avg_3, np.insert(pknag_indexes[:,2], 0, t_0), ch_picks, True, info_obj)
[times, pca_tcuts, pca_coeffs, pca_expVar] = modTRFSources(t_epochs, Ht_avg_4, np.insert(pknag_indexes[:,3], 0, t_0), ch_picks, True, info_obj)


#%% Spectrum of mod-TRF
t_0 = np.where(t_epochs >=0)[0][0]
t_500 = np.where(t_epochs >=0.5)[0][0]
fs=4096.

Ht1_fft = np.fft.fft(Ht_avg_1[t_0:t_500,31] - Ht_avg_1[t_0:t_500,31].mean())
Ht2_fft = np.fft.fft(Ht_avg_2[t_0:t_500,31] - Ht_avg_2[t_0:t_500,31].mean())
Ht3_fft = np.fft.fft(Ht_avg_3[t_0:t_500,31] - Ht_avg_3[t_0:t_500,31].mean())
Ht4_fft = np.fft.fft(Ht_avg_4[t_0:t_500,31] - Ht_avg_4[t_0:t_500,31].mean())
f = np.fft.fftfreq(t_500-t_0,d=1/fs)

plt.figure()
plt.plot(f,np.abs(Ht1_fft))
plt.plot(f,np.abs(Ht2_fft))
plt.plot(f,np.abs(Ht3_fft))
plt.plot(f,np.abs(Ht4_fft))
plt.legend(['Young', 'Less Young', 'MA', 'Old'])
plt.xlim([0,150])



#%% modTRF feature extraction in individuals

for s in range(len(Subjects)):
    subject = Subjects[s]
    Ht = A_Hts[s].T
    ch_picks_s = A_chPicks[s]
    
    Ht = central_gain_modTRF(t_epochs, Ht)
    [pk_indexes, pkn_indexes, pk_latencys, pk_widths, x, x, x] = peakAnalysis(t_epochs, Ht, ch_picks_s, False)
    
    ch_cz = Ht[:, ch_picks_s==31]
    
    plt.figure()
    plt.plot(t_epochs, ch_cz)
    plt.plot(t_epochs[pk_indexes], ch_cz[pk_indexes], 'x')
    plt.title('Cz')
    plt.xlim((-0.05, 0.5))
    plt.savefig(os.path.join(fig_loc, 'ModTRF_pksLabeled', subject + '_modTRF_PeaksCz.png'),format='png')
    plt.close()
    
    
    
#%% Central gain extraction 

Cg1 = np.zeros(len(Subjects)) #Central gain computed with peak 1
Cg2 = np.zeros(len(Subjects)) #Central gain computed with peak 2

for s in range(len(Subjects)):
    subject = Subjects[s]
    Ht = A_Hts[s].T
    ch_picks_s = A_chPicks[s]
    
    Ht = central_gain_modTRF(t_epochs, Ht)
    ch_cz = Ht[:, ch_picks_s==31].squeeze()
    
    maxlocs = TwoPeaks(t_epochs, ch_cz)
    
    plt.figure()
    plt.plot(t_epochs, ch_cz)
    plt.plot(t_epochs[maxlocs], ch_cz[maxlocs], 'x')
    plt.title('Cz')
    plt.xlim((-0.05, 0.5))
    plt.savefig(os.path.join(fig_loc, 'ModTRF_maxPks', subject + '_modTRF_MaxPeaksCz.png'),format='png')
    plt.close()
    
    Cg1[s] = ch_cz[maxlocs[0]]
    Cg2[s] = ch_cz[maxlocs[1]]
    

plt.figure()
plt.plot(Cg1,Cg2,'x')

#%% Central gain vs age and behavior

plt.figure()
plt.plot(age,Cg1,'x', label='CG first 2 sources')
plt.plot(age,Cg2,'x', label = 'CG last 2 sources')
plt.xlabel('Age')
plt.ylabel('Central Gain')
plt.legend()
  

jane_cg1 = Cg1[index_sub_jane]
jane_cg2 = Cg2[index_sub_jane]
 
plt.figure()
plt.plot(jane_cg1, jane_thresh,'x', label='CG first 2 sources')
plt.plot(jane_cg2, jane_thresh,'x', label='CG last 2 sources')
plt.xlabel('Central Gain')
plt.ylabel('Jane_thresh')
plt.legend()

mrt_cg1 = Cg1[index_sub_mrt]
mrt_cg2 = Cg2[index_sub_mrt]

plt.figure()
plt.plot(mrt_cg1, mrt_thresh,'x', label='CG first 2 sources')
plt.plot(mrt_cg2, mrt_thresh,'x', label='CG last 2 sources')
plt.xlabel('Central Gain')
plt.ylabel('MRT thresh')
plt.legend()

r_age1 = pearsonr(np.delete(age,np.argmax(Cg1)), np.delete(Cg1,np.argmax(Cg1)))
r_age2 = pearsonr(age, Cg2)

r_jane1 = pearsonr(jane_cg1, jane_thresh)
r_jane2 = pearsonr(jane_cg2, jane_thresh)

r_mrt1 = pearsonr(mrt_cg1, mrt_thresh)
r_mrt2 = pearsonr(mrt_cg2, mrt_thresh)

r_vals = [r_age1[0], r_age2[0], r_jane1[0], r_jane2[0], r_mrt1[0], r_mrt2[0]]
p_vals = [r_age1[1], r_age2[1], r_jane1[1], r_jane2[1], r_mrt1[1], r_mrt2[0]]







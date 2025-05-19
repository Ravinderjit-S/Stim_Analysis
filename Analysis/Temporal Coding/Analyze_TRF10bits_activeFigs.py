#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 14:12:16 2022

@author: ravinderjit

make figues to show what happens to mod-TRF under attention

"""


import os
import pickle
import numpy as np
import mne
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import scipy.io as sio
from scipy.signal import find_peaks
from scipy.stats import ttest_rel


#%% Load mseq
mseq_loc = '/media/ravinderjit/Data_Drive/Data/EEGdata/TemporalCoding/mseqEEG_150_bits10_4096.mat'
Mseq_dat = sio.loadmat(mseq_loc)
mseq = Mseq_dat['mseqEEG_4096'].astype(float)
        
#%% Subjects

Subjects = ['S207', 'S211', 'S228','S236','S238','S239','S246','S247','S250']
#Exclude subjects S246 and S247 as they didn't do hard task
Subjects = ['S207', 'S211', 'S228','S236','S238','S239','S250']
Subjects_sd = ['S207', 'S211', 'S228', 'S236', 'S238', 'S239', 'S250'] 


dataPassive_loc = '/media/ravinderjit/Data_Drive/Data/EEGdata/TemporalCoding/AMmseq_10bits/'
picklePassive_loc = dataPassive_loc + 'Pickles/'

dataCount_loc = '/media/ravinderjit/Data_Drive/Data/EEGdata/TemporalCoding/AMmseq_active/'
pickleCount_loc = dataCount_loc + 'Pickles/'

pickle_loc_sd = '/media/ravinderjit/Data_Drive/Data/EEGdata/TemporalCoding/AMmseq_active_harder/Pickles/'

fig_path = os.path.abspath('/media/ravinderjit/Data_Drive/Data/Figures/TemporalCoding/')

#%% Norm factor
#This is the normalization factor to account for the extension of the mseq to the emseq
cof = 150
T = 1/150 #this is the amount each point is extended
fs = 4096
norm_fac = np.floor(fs*T)


#%% Load Passive Data

A_Tot_trials_pass = []
A_Ht_pass = []
A_Htnf_pass = []
A_info_obj_pass = []
A_ch_picks_pass = []

A_Ht_epochs_pass = []

for sub in range(len(Subjects)):
    print('Loading ' + Subjects[sub])
    subject = Subjects[sub]
    if subject == 'S250':
        subject = 'S250_visit2'
    with open(os.path.join(picklePassive_loc,subject +'_AMmseq10bits.pickle'),'rb') as file:
        [t, Tot_trials, Ht, Htnf, info_obj, ch_picks] = pickle.load(file)
        
    Ht = Ht / norm_fac
    Htnf = Htnf / norm_fac
        
    A_Tot_trials_pass.append(Tot_trials)
    A_Ht_pass.append(Ht)
    A_Htnf_pass.append(Htnf)
    A_info_obj_pass.append(info_obj)
    A_ch_picks_pass.append(ch_picks)
    
    with open(os.path.join(picklePassive_loc,subject +'_AMmseq10bits_epochs.pickle'),'rb') as file:
        [Ht_epochs, t_epochs] = pickle.load(file)
    
    Ht_epochs = Ht_epochs / norm_fac
    
    A_Ht_epochs_pass.append(Ht_epochs)

print('Done loading passive ...')

#%% Load Counting Data

A_Tot_trials_count = []
A_Ht_count = []
A_Htnf_count = []
A_info_obj_count = []
A_ch_picks_count = []

A_Ht_epochs_count = []

for sub in range(len(Subjects)):
    subject = Subjects[sub]
    print ('loading ' + subject)
    with open(os.path.join(pickleCount_loc,subject +'_AMmseq10bits_Active.pickle'),'rb') as file:
        [t, Tot_trials, Ht, Htnf, info_obj, ch_picks] = pickle.load(file)
        
    Ht = Ht / norm_fac
    Htnf = Htnf / norm_fac
        
    A_Tot_trials_count.append(Tot_trials)
    A_Ht_count.append(Ht)
    A_Htnf_count.append(Htnf)
    A_info_obj_count.append(info_obj)
    A_ch_picks_count.append(ch_picks)
    
    
    with open(os.path.join(pickleCount_loc,subject +'_AMmseq10bits_Active_epochs.pickle'),'rb') as file:
        [Ht_epochs, t_epochs] = pickle.load(file)
        
    Ht_epochs = Ht_epochs / norm_fac
        
    A_Ht_epochs_count.append(Ht_epochs)
    
print('Done Loading Counting data')

#%% Load Shift Detect Data

Subjects_sd = ['S207', 'S211', 'S228', 'S236', 'S238', 'S239', 'S250'] 

A_Tot_trials_sd = []
A_Ht_sd = []
A_info_obj_sd = []
A_ch_picks_sd = []

A_Ht_epochs_sd = []

for sub in range(len(Subjects_sd)):
    subject = Subjects_sd[sub]
    print('loading ' + subject)
    with open(os.path.join(pickle_loc_sd,subject +'_AMmseq10bit_Active_harder.pickle'),'rb') as file:
        [t, Tot_trials, Ht, info_obj, ch_picks] = pickle.load(file)
        
    Ht = Ht / norm_fac
    Htnf = Htnf / norm_fac
    
    A_Tot_trials_sd.append(Tot_trials)
    A_Ht_sd.append(Ht)
    A_info_obj_sd.append(info_obj)
    A_ch_picks_sd.append(ch_picks)
    
    with open(os.path.join(pickle_loc_sd,subject +'_AMmseq10bit_Active_harder_epochs.pickle'),'rb') as file:
        [Ht_epochs, t_epochs] = pickle.load(file)
        
    Ht_epochs = Ht_epochs / norm_fac
        
    A_Ht_epochs_sd.append(Ht_epochs)
    
print('Done loading shift detect data')


#%% Plot Ch. Cz

t_epochs *= 1e3 #change to ms    
fig,ax = plt.subplots(nrows=2,ncols=2,sharex=True)
fig.set_size_inches(12,8)
ax = np.reshape(ax,4)

t_0 = np.where(t_epochs>=0)[0][0]

for sub in range(len(Subjects[:4])):
    subject = Subjects[sub]
    
    ch = 31
    
    #if sub !=2:
        #ax[sub].axes.yaxis.set_visible(False)
        
    # if sub < len(Subjects)/2:
    #     ax[sub].axes.xaxis.set_visible(False)
    
    ch_pass_ind = np.where(A_ch_picks_pass[sub] == ch)[0][0]
    ch_count_ind = np.where(A_ch_picks_count[sub] == ch)[0][0]
    
    cz_pass = A_Ht_epochs_pass[sub][ch_pass_ind,:,:] *1e6
    cz_count = A_Ht_epochs_count[sub][ch_count_ind,:,:] * 1e6
    
    cz_pass_sem = cz_pass.std(axis=0) / np.sqrt(cz_pass.shape[0])
    cz_count_sem = cz_count.std(axis=0) / np.sqrt(cz_count.shape[0])
    
    cz_pass = cz_pass.mean(axis=0)
    cz_count = cz_count.mean(axis=0)
    
    cz_pass = cz_pass - cz_pass[t_0] #make time 0 have value of 0
    cz_count = cz_count - cz_count[t_0] #make time 0 have value of 0
    
    ax[sub].plot(t_epochs,cz_pass, label='Passive', color='k',linewidth=2)
    ax[sub].fill_between(t_epochs,cz_pass-cz_pass_sem, cz_pass+cz_pass_sem,color='k',alpha=0.5)
    
    ax[sub].plot(t_epochs,cz_count, label='Count (easy)', color='tab:blue')
    ax[sub].fill_between(t_epochs,cz_count-cz_count_sem, cz_count+cz_count_sem,color='tab:blue',alpha=0.5)
    
    if (subject in Subjects_sd):
        sub_sd = Subjects_sd.index(subject)
        
        ch_sd_ind = np.where(A_ch_picks_sd[sub_sd] == ch)[0][0]
        cz_sd = A_Ht_epochs_sd[sub_sd][ch_sd_ind,:,:] * 1e6
        
        cz_sd_sem = cz_sd.std(axis=0) / np.sqrt(cz_sd.shape[0])
        cz_sd = cz_sd.mean(axis=0)
        
        cz_sd = cz_sd - cz_sd[t_0]
        
        ax[sub].plot(t_epochs,cz_sd, label='Shift Detect (hard)', color='tab:orange')
        ax[sub].fill_between(t_epochs,cz_sd - cz_sd_sem, cz_sd + cz_sd_sem, color='tab:orange',alpha=0.5)
    
    #ax[sub].set_title('S' + str(sub+1))
    ax[sub].set_title('P' + str(sub+1),fontweight='bold',fontsize=14)
    ax[sub].set_xlim([-0.010*1e3,0.3*1e3])
    #ax[sub].set_xlim([-.005,0.050])
    #ax[sub].set_xticks([0,0.050,0.1])
    #ax[sub].set_xticks([0,0.2,0.4])
    
ax[2].legend(fontsize=12)
#ax[2].axes.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
ax[2].set_xlabel('Time (msec)',fontsize=14)
ax[2].set_ylabel('\u03BCV',fontsize=14)
# ax[0].set_yticks([-15,0,15,30])
# ax[1].set_yticks([-20,-10,0,10])
# ax[2].set_yticks([-30,0,30,60])
# ax[3].set_yticks([-40,-20,0,20])
for a_ in ax:
    a_.tick_params(axis='both', labelsize=14)

plt.savefig(os.path.join(fig_path,'ModTRF_active.svg'),format='svg')

t_epochs /= 1e3

#%% Average Attention Effects

# Average Passive Data

t_neg = np.where(t>=-0.007)[0][0] # baseline from the last 7 ms
t_0 = np.where(t>=0)[0][0]

#32 ch average
perCh_pass = np.zeros([32,1])
for s in range(len(Subjects)):
    for ch in range(32):
        perCh_pass[ch,0] += np.sum(A_ch_picks_pass[s]==ch)

Avg_Ht_pass = np.zeros([32,t.size])
Ht_pass_cz = np.zeros([len(Subjects),t.size])

for s in range(len(Subjects)):
    Ht_s = A_Ht_pass[s] - A_Ht_pass[s][:,t_neg:t_0].mean(axis=1)[:,np.newaxis] #make time zero = 0
    Avg_Ht_pass[A_ch_picks_pass[s],:] += Ht_s
    Ht_pass_cz[s,:] = Ht_s[-1,:]

Avg_Ht_pass = Avg_Ht_pass / perCh_pass

# Average Counting Data
#32 ch average
perCh_count = np.zeros([32,1])
for s in range(len(Subjects)):
    for ch in range(32):
        perCh_count[ch,0] += np.sum(A_ch_picks_count[s]==ch)

Avg_Ht_count = np.zeros([32,t.size])
Ht_count_cz = np.zeros([len(Subjects),t.size])

for s in range(len(Subjects)):
    Ht_s = A_Ht_count[s] - A_Ht_count[s][:,t_neg:t_0].mean(axis=1)[:,np.newaxis] #make time zero = 0
    Avg_Ht_count[A_ch_picks_count[s],:] += Ht_s
    Ht_count_cz[s,:] = Ht_s[-1,:]

Avg_Ht_count = Avg_Ht_count / perCh_count

# Average Shift Detect Data 
#32 ch average
perCh_sd = np.zeros([32,1])
for s in range(len(Subjects_sd)):
    for ch in range(32):
        perCh_sd[ch,0] += np.sum(A_ch_picks_sd[s]==ch)

Avg_Ht_sd = np.zeros([32,t.size])
Ht_sd_cz = np.zeros([len(Subjects_sd),t.size])

for s in range(len(Subjects_sd)):
    Ht_s = A_Ht_sd[s] - A_Ht_sd[s][:,t_neg:t_0].mean(axis=1)[:,np.newaxis] #make time zero = 0
    Avg_Ht_sd[A_ch_picks_sd[s],:] += Ht_s
    Ht_sd_cz[s,:] = Ht_s[-1,:]

Avg_Ht_sd = Avg_Ht_sd / perCh_sd

plt.figure()
plt.plot(t, Avg_Ht_pass[31,:])
plt.plot(t, Avg_Ht_count[31,:])
plt.plot(t, Avg_Ht_sd[31,:])
plt.xlim([-0.05,0.4])

pass_mean = Ht_pass_cz.mean(axis=0) * 1e6
count_mean = Ht_count_cz.mean(axis=0)* 1e6
sd_mean = Ht_sd_cz.mean(axis=0)* 1e6

pass_sem = Ht_pass_cz.std(axis=0) / np.sqrt(Ht_pass_cz.shape[0])* 1e6
count_sem = Ht_count_cz.std(axis=0) / np.sqrt(Ht_count_cz.shape[0])* 1e6
sd_sem = Ht_sd_cz.std(axis=0) / np.sqrt(Ht_sd_cz.shape[0])* 1e6

fig = plt.figure()
t*=1e3
fig.set_size_inches(10,5)
plt.plot(t,pass_mean,color='k',linewidth=2, label='Passive')
plt.fill_between(t,pass_mean - pass_sem, pass_mean+pass_sem,color='k',alpha=0.5)

plt.plot(t,count_mean, color='tab:blue',linewidth=2, label='Easy (Counting)')
plt.fill_between(t,count_mean - count_sem, count_mean+pass_sem,color='tab:blue',alpha=0.5)

plt.plot(t, sd_mean, color='tab:orange',linewidth=2, label='Hard (Shift Detect)')
plt.fill_between(t,sd_mean - sd_sem, sd_mean+ sd_sem,color='tab:orange',alpha=0.5)
t/=1e3
plt.xlim(-50,300)
#plt.xticks([0,50,100,200,300])
#plt.yticks([-15,0,15,30])
plt.ylabel('\u03BCV',fontsize=14)
plt.xlabel('Time (msec)',fontsize=14)
plt.tick_params(labelsize=14)
plt.legend(fontsize=12)

plt.savefig(os.path.join(fig_path,'ModTRF_active_avg.svg'),format='svg')

#Same Figure but zoom in on first 50 ms

fig = plt.figure()
t*=1e3
fig.set_size_inches(12,7)
plt.plot(t,pass_mean,color='k',linewidth=2, label='Passive')
plt.fill_between(t,pass_mean - pass_sem, pass_mean+pass_sem,color='k',alpha=0.5)

plt.plot(t,count_mean, color='tab:blue',linewidth=2, label='Easy (Counting)')
plt.fill_between(t,count_mean - count_sem, count_mean+pass_sem,color='tab:blue',alpha=0.5)

plt.plot(t, sd_mean, color='tab:orange',linewidth=2, label='Hard (Shift Detect)')
plt.fill_between(t,sd_mean - sd_sem, sd_mean+ sd_sem,color='tab:orange',alpha=0.5)
t/=1e3
plt.xlim(-10,50)
#plt.xticks([0,50,100,200,300])
#plt.yticks([-15,0,15,30])
plt.ylabel('\u03BCV',fontsize=22)
plt.xlabel('Time (msec)',fontsize=22)
plt.tick_params(labelsize=22)
#plt.legend()

plt.savefig(os.path.join(fig_path,'ModTRF_active_avg_1st50.svg'),format='svg')



#%% PCA on Average Attention Effects to get rough sources

t_cuts_pass = [.016, .033, .066, .123, .268 ]
t_cuts_count = [.015, .037, .068, .120, .260  ]
t_cuts_sd = [.013, .036, .066, .123, .260 ]


pca_sp_pass = []
pca_expVar_pass = []
pca_coeff_pass = []
t_cutT_pass = []

pca_sp_count = []
pca_expVar_count = []
pca_coeff_count = []
t_cutT_count = []

pca_sp_sd = []
pca_expVar_sd = []
pca_coeff_sd = []
t_cutT_sd = []

pca = PCA(n_components=1)

for t_c in range(len(t_cuts_pass)):
    
    # Get times for t_cut
    if t_c ==0:
        t_1_p = np.where(t>=0)[0][0]
        t_1_c = t_1_p
        t_1_s = t_1_p
    else:
        t_1_p = np.where(t>=t_cuts_pass[t_c-1])[0][0]
        t_1_c = np.where(t>=t_cuts_count[t_c-1])[0][0]
        t_1_s = np.where(t>=t_cuts_sd[t_c-1])[0][0]
         
    t_2_p = np.where(t>=t_cuts_pass[t_c])[0][0]
    t_2_c = np.where(t>=t_cuts_count[t_c])[0][0]
    t_2_s = np.where(t>=t_cuts_sd[t_c])[0][0]
    
    #PCA
    pca_sp_pass_tc = pca.fit_transform(Avg_Ht_pass[:,t_1_p:t_2_p].T)
    pca_expVar_pass_tc = pca.explained_variance_ratio_
    pca_coeff_pass_tc = pca.components_
    
    pca_sp_count_tc = pca.fit_transform(Avg_Ht_count[:,t_1_p:t_2_p].T)
    pca_expVar_count_tc = pca.explained_variance_ratio_
    pca_coeff_count_tc = pca.components_
    
    pca_sp_sd_tc = pca.fit_transform(Avg_Ht_sd[:,t_1_p:t_2_p].T)
    pca_expVar_sd_tc = pca.explained_variance_ratio_
    pca_coeff_sd_tc = pca.components_
    
    
    # Get polarity with positive at Cz
    if pca_coeff_pass_tc[0,31] < 0:  # Expand this too look at mutlitple electrodes
       pca_coeff_pass_tc = -pca_coeff_pass_tc
       pca_sp_pass_tc = -pca_sp_pass_tc
       
    if pca_coeff_count_tc[0,31] < 0:  # Expand this too look at mutlitple electrodes
       pca_coeff_count_tc = -pca_coeff_count_tc
       pca_sp_count_tc = -pca_sp_count_tc
    
    if pca_coeff_sd_tc[0,31] < 0:  # Expand this too look at mutlitple electrodes
       pca_coeff_sd_tc = -pca_coeff_sd_tc
       pca_sp_sd_tc = -pca_sp_sd_tc
       
    #Store in lists
    pca_sp_pass.append(pca_sp_pass_tc)
    pca_expVar_pass.append(pca_expVar_pass_tc)
    pca_coeff_pass.append(pca_coeff_pass_tc)
    t_cutT_pass.append(t[t_1_p:t_2_p])
    
    pca_sp_count.append(pca_sp_count_tc)
    pca_expVar_count.append(pca_expVar_count_tc)
    pca_coeff_count.append(pca_coeff_count_tc)
    t_cutT_count.append(t[t_1_p:t_2_p])
    
    pca_sp_sd.append(pca_sp_sd_tc)
    pca_expVar_sd.append(pca_expVar_sd_tc)
    pca_coeff_sd.append(pca_coeff_sd_tc)
    t_cutT_sd.append(t[t_1_p:t_2_p])



plt.figure()
for t_c in range(len(t_cuts_pass)):
    plt.plot(t_cutT_pass[t_c],pca_sp_pass[t_c][:,0])
plt.plot(t,Avg_Ht_pass[31,:] -Avg_Ht_pass[31,:].mean(),color='k')
plt.xlim([0,0.5])
    

fig = plt.figure()
fig.set_size_inches(10,5)
labels = ['comp1']
vmin = pca_coeff_pass[-1].mean() - 2 * pca_coeff_pass[-1].std()
vmax = pca_coeff_pass[-1].mean() + 2 * pca_coeff_pass[-1].std()
for t_c in range(len(t_cuts_pass)):
    ax = plt.subplot(3,len(t_cuts_pass),t_c+1)
    plt.title( str(np.round(pca_expVar_pass[t_c][0]*100)) + '%',horizontalalignment='right')
    mne.viz.plot_topomap(pca_coeff_pass[t_c][0,:], mne.pick_info(info_obj,A_ch_picks_pass[1]),vlim=(vmin,vmax),axes=ax)
    
    ax = plt.subplot(3,len(t_cuts_pass),t_c+1 + len(t_cuts_pass))
    plt.title(str(np.round(pca_expVar_count[t_c][0]*100)) + '%',horizontalalignment='right')
    mne.viz.plot_topomap(pca_coeff_count[t_c][0,:], mne.pick_info(info_obj,A_ch_picks_pass[1]),vlim=(vmin,vmax),axes=ax)
    
    ax = plt.subplot(3,len(t_cuts_pass),t_c+1 + 2* len(t_cuts_pass))
    plt.title( str(np.round(pca_expVar_sd[t_c][0]*100)) + '%',horizontalalignment='right')
    mne.viz.plot_topomap(pca_coeff_sd[t_c][0,:], mne.pick_info(info_obj,A_ch_picks_pass[1]),vlim=(vmin,vmax),axes=ax)
    
plt.savefig(os.path.join(fig_path,'topomaps.svg'),format='svg')


#%% Extract peak amplitudes and latencies for stats

#################
#Acheived through use of find_peaks and manual insepction

sub_p = 6

t_0 = np.where(t>=0)[0][0]
t_1 = np.where(t>=0.3)[0][0]
pk_locs_pass, props = find_peaks(-Ht_pass_cz[sub_p,t_0:t_1])
pk_locs_pass = pk_locs_pass+t_0

pk_locs_count, props = find_peaks(-Ht_count_cz[sub_p,t_0:t_1])
pk_locs_count = pk_locs_count+t_0

pk_locs_sd, props = find_peaks(-Ht_sd_cz[sub_p,t_0:t_1])
pk_locs_sd = pk_locs_sd+t_0

plt.figure()
plt.plot(t,Ht_pass_cz[sub_p,:],color='k')
plt.scatter(t[pk_locs_pass],Ht_pass_cz[sub_p,pk_locs_pass],color='grey', marker='x', s=50)
plt.plot(t,Ht_count_cz[sub_p,:], color='tab:blue')
plt.scatter(t[pk_locs_count],Ht_count_cz[sub_p,pk_locs_count],color='purple', marker='x', s=50)
plt.plot(t,Ht_sd_cz[sub_p,:],color='tab:red')
plt.scatter(t[pk_locs_sd],Ht_sd_cz[sub_p,pk_locs_sd],color='pink', marker='x', s=50)
plt.xlim([-0.1,0.3])
plt.title(Subjects[sub_p])

#################

pk_locs_pass_pos = np.array((
    [27921, 28008, 28083, 28267, 28707],
    [27920, 28012, 28093, 28233, 28797],
    [27921, 27993, 28127, 28269, 28666],
    [27916, 28056, 28136, 28250, 28627],
    [27918, 28020, 28108, 28274, 28614],
    [27920, 28003, 28080, 28242, 28568],
    [27919, 28001, 28218, 28578, 28976],
    ), dtype=int)

pk_locs_count_pos = np.array((
    [27920, 28002, 28079, 28276, 28692],
    [27919, 27986, 28074, 28227, 28703],
    [27919, 27986, 28130, 28262, 28582],
    [27916, 28047, 28107, 28268, 28630],
    [27917, 28004, 28097, 28281, 28541],
    [27917, 27989, 28072, 28265, 28638],
    [27916, 27994, 28178, 28512, 29025],
    ), dtype=int)

pk_locs_sd_pos = np.array((
    [27912, 27994, 28074, 28276, 28652],
    [27912, 28007, 28090, 28216, 28750],
    [27910, 27981, 28115, 28268, 28620],
    [27906, 27987, 28116, 28262, 28603],
    [27913, 27995, 28108, 28305, 28462],
    [27908, 27980, 28066, 28297, 28562],
    [27909, 27992, 28197, 28572, 28969],
    
    ), dtype=int)

pk_locs_pass_neg = np.array((
    [27971, 28036, 28170, 28452],
    [27956, 28050, 28151, 28389],
    [27957, 28081, 28172, 28408],
    [27949, 28116, 28162, 28391],
    [27957, 28102, 28161, 28403],
    [27956, 28037, 28150, 28376],
    [27955, 28106, 28397, 28769]
    ), dtype=int)

pk_locs_count_neg = np.array((
    [27964, 28030, 28172, 28445],
    [27952, 28060, 28131, 28365],
    [27950, 28064, 28177, 28415],
    [27949, 28095, 28170, 28403],
    [27951, 28088, 28174, 28379],
    [27948, 28035, 28167, 28346],
    [27950, 28094, 28393, 28728],
    ), dtype=int)

pk_locs_sd_neg = np.array((
    [27949, 28024, 28164, 28386],
    [27952, 28044, 28134, 28358],
    [27943, 28055, 28164, 28388],
    [27945, 28087, 28168, 28395],
    [27949, 28075, 28168, 28361],
    [27941, 28028, 28146, 28371],
    [27947, 28086, 28406, 28789]
    ), dtype=int)


plt.figure()
plt.plot(t,Ht_pass_cz[sub_p,:],color='k')
plt.scatter(t[pk_locs_pass_pos[sub_p,:]],Ht_pass_cz[sub_p,pk_locs_pass_pos[sub_p,:]],color='grey', marker='x', s=50)
plt.scatter(t[pk_locs_pass_neg[sub_p,:]],Ht_pass_cz[sub_p,pk_locs_pass_neg[sub_p,:]],color='grey', marker='x', s=50)
plt.plot(t,Ht_count_cz[sub_p,:], color='tab:blue')
plt.scatter(t[pk_locs_count_pos[sub_p,:]],Ht_count_cz[sub_p,pk_locs_count_pos[sub_p,:]],color='purple', marker='x', s=50)
plt.scatter(t[pk_locs_count_neg[sub_p,:]],Ht_count_cz[sub_p,pk_locs_count_neg[sub_p,:]],color='purple', marker='x', s=50)
plt.plot(t,Ht_sd_cz[sub_p,:],color='tab:red')
plt.scatter(t[pk_locs_sd_pos[sub_p,:]],Ht_sd_cz[sub_p,pk_locs_sd_pos[sub_p,:]],color='pink', marker='x', s=50)
plt.scatter(t[pk_locs_sd_neg[sub_p,:]],Ht_sd_cz[sub_p,pk_locs_sd_neg[sub_p,:]],color='pink', marker='x', s=50)
plt.xlim([-0.1,0.3])
plt.title(Subjects[sub_p])

##Pk amplitudes
pk1_amps = np.zeros([7,3])
n1_amps = np.zeros([7,3])

pk2_amps = np.zeros([7,3])
n2_amps = np.zeros([7,3])

pk5_amps = np.zeros([7,3])

for sub_p in range(len(Subjects)):
    pk1_amps[sub_p,0] = Ht_pass_cz[sub_p,pk_locs_pass_pos[sub_p,0]]
    pk1_amps[sub_p,1] = Ht_count_cz[sub_p,pk_locs_count_pos[sub_p,0]]
    pk1_amps[sub_p,2] = Ht_sd_cz[sub_p,pk_locs_sd_pos[sub_p,0]]
    
    n1_amps[sub_p,0] = Ht_pass_cz[sub_p,pk_locs_pass_neg[sub_p,0]]
    n1_amps[sub_p,1] = Ht_count_cz[sub_p,pk_locs_count_neg[sub_p,0]]
    n1_amps[sub_p,2] = Ht_sd_cz[sub_p,pk_locs_sd_neg[sub_p,0]]
    
    pk2_amps[sub_p,0] = Ht_pass_cz[sub_p,pk_locs_pass_pos[sub_p,1]]
    pk2_amps[sub_p,1] = Ht_count_cz[sub_p,pk_locs_count_pos[sub_p,1]]
    pk2_amps[sub_p,2] = Ht_sd_cz[sub_p,pk_locs_sd_pos[sub_p,1]]
    
    n2_amps[sub_p,0] = Ht_pass_cz[sub_p,pk_locs_pass_neg[sub_p,1]]
    n2_amps[sub_p,1] = Ht_count_cz[sub_p,pk_locs_count_neg[sub_p,1]]
    n2_amps[sub_p,2] = Ht_sd_cz[sub_p,pk_locs_sd_neg[sub_p,1]]
    
    pk5_amps[sub_p,0] = Ht_pass_cz[sub_p,pk_locs_pass_pos[sub_p,4]]
    pk5_amps[sub_p,1] = Ht_count_cz[sub_p,pk_locs_count_pos[sub_p,4]]
    pk5_amps[sub_p,2] = Ht_sd_cz[sub_p,pk_locs_sd_pos[sub_p,4]]
    
    

##Test for statistical differences
#Pk 1
ttest_p1_1 = ttest_rel(t[pk_locs_pass_pos[:,0]],t[pk_locs_count_pos[:,0]],alternative='greater' ) #Passive vs count
avg_diff_p1_pass_count =  np.mean(t[pk_locs_pass_pos[:,0]] - t[pk_locs_count_pos[:,0]])

print('Pk1_pass vs Pk1_count diff: ', avg_diff_p1_pass_count*1e3 , 'ms | p-val = ', ttest_p1_1[1], '\n')


ttest_p1_2 =ttest_rel(t[pk_locs_pass_pos[:,0]],t[pk_locs_sd_pos[:,0]],alternative='greater' ) #Passive vs SD
avg_diff_p1_pass_sd =  np.mean(t[pk_locs_pass_pos[:,0]] - t[pk_locs_sd_pos[:,0]])

print('Pk1_pass vs Pk1_sd diff: ', avg_diff_p1_pass_sd*1e3, 'ms | p-val = ', ttest_p1_2[1], '\n')

ttest_p1_amp1 = ttest_rel(pk1_amps[:,0],pk1_amps[:,2], alternative='greater')
(pk1_amps[:,0].mean() - pk1_amps[:,2].mean()) * 1e6

print('Amp pass_sd diff p-val: ', ttest_p1_amp1[1])

#Neg Pk 1
ttest_n1_1 = ttest_rel(t[pk_locs_pass_neg[:,0]],t[pk_locs_sd_neg[:,0]],alternative='greater' ) #Passive vs count
avg_diff_pk1_neg_pass_sd =  np.mean(t[pk_locs_pass_neg[:,0]] - t[pk_locs_sd_neg[:,0]])

ttest_n1_2 =ttest_rel(t[pk_locs_count_neg[:,0]],t[pk_locs_sd_neg[:,0]],alternative='greater' ) #Passive vs SD
avg_diff_pk1_neg_count_sd =  np.mean(t[pk_locs_count_neg[:,0]] - t[pk_locs_sd_neg[:,0]])

ttest_pkn1_amp1 = ttest_rel(n1_amps[:,0],n1_amps[:,1], alternative='greater')
ttest_pkn1_amp2 = ttest_rel(n1_amps[:,0],n1_amps[:,2], alternative='greater')

#Pos Pk 2
ttest_p2_1 = ttest_rel(t[pk_locs_pass_pos[:,1]],t[pk_locs_count_pos[:,1]],alternative='greater' ) #Passive vs count
avg_diff_pk1_pass_count =  np.mean(t[pk_locs_pass_pos[:,1]] - t[pk_locs_count_pos[:,1]])

ttest_p2_2 =ttest_rel(t[pk_locs_pass_pos[:,1]],t[pk_locs_sd_pos[:,1]],alternative='greater' ) #Passive vs SD
avg_diff_pk1_pass_sd =  np.mean(t[pk_locs_pass_pos[:,1]] - t[pk_locs_sd_pos[:,1]])

ttest_pk1_amp1 = ttest_rel(pk2_amps[:,0],pk2_amps[:,2], alternative='less')
ttest_pk1_amp2 = ttest_rel(pk2_amps[:,0],pk2_amps[:,1], alternative='less')

(pk2_amps[:,2].mean() - pk2_amps[:,0].mean())*1e6
(pk2_amps[:,1].mean() - pk2_amps[:,0].mean())*1e6

#Neg Pk 2
ttest_n2_1 = ttest_rel(t[pk_locs_pass_neg[:,1]],t[pk_locs_sd_neg[:,1]],alternative='greater' ) 
avg_diff_n2_pass_count = np.mean(t[pk_locs_pass_neg[:,1]] - t[pk_locs_count_neg[:,1]])

ttest_n2_2 =ttest_rel(t[pk_locs_count_neg[:,1]],t[pk_locs_sd_neg[:,1]],alternative='greater' ) #Passive vs SD
avg_diff_pk1_neg_count_sd =  np.mean(t[pk_locs_count_neg[:,0]] - t[pk_locs_sd_neg[:,0]])

ttest_pkn2_amp1 = ttest_rel(n2_amps[:,0],n2_amps[:,1], alternative='greater')

#Pos Pk 5
ttest_p5_1 = ttest_rel(t[pk_locs_pass_pos[:,4]],t[pk_locs_count_pos[:,4]],alternative='greater' ) #Passive vs count
avg_diff_p5_pass_count =  np.mean(t[pk_locs_pass_pos[:,4]] - t[pk_locs_count_pos[:,4]])

print('Pk1_pass vs Pk1_count diff: ', avg_diff_p1_pass_count*1e3 , 'ms | p-val = ', ttest_p1_1[1], '\n')


ttest_p5_2 = ttest_rel(t[pk_locs_pass_pos[:,4]],t[pk_locs_sd_pos[:,4]],alternative='greater' ) #Passive vs count
avg_diff_p5_pass_sd =  np.mean(t[pk_locs_pass_pos[:,4]] - t[pk_locs_sd_pos[:,4]])

ttest_pk5_amp1 = ttest_rel(pk5_amps[:,0],pk5_amps[:,1], alternative='greater')
ttest_pk5_amp2 = ttest_rel(pk5_amps[:,0],pk5_amps[:,2], alternative='greater')





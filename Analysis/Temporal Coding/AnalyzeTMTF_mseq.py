# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
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

nchans = 34;
refchans = ['EXG1','EXG2']

data_eeg = [];
data_evnt = [];


data_loc = '/media/ravinderjit/Data_Drive/Data/EEGdata/TemporalCoding/TMTF/'
pickle_loc = data_loc + 'Pickles/'

Subjects = ['E001_1', 'E001_2','E002_Visit_1','E002_Visit_2','E002_Visit_3',
            'E003', 'E004_Visit_1','E004_Visit_2','E004_Visit_3', 'E005_Visit_1',
            'E005_Visit_2', 'E006_Visit_1', 'E006_Visit_2', 'E006_Visit_3',
            'E007_Visit_1', 'E007_Visit_2', 'E007_Visit_3', 'E012_Visit_1', 'E012_Visit_2' ,
            'E012_Visit_3', 'E014', 'E016', 'E022_Visit_1', 'E022_Visit_2', 'E022_Visit_3',
            ]


# E007_Visit_2 has weird blink projection
# ask about E006_Visit1_R+004

mseq_loc = os.path.join(data_loc, 'TMTFmseq_resampled.mat')
Mseq_dat = sio.loadmat(mseq_loc)
mseq = Mseq_dat['m'].astype('float')
mseq = 2*mseq - 1
mseq[mseq<0] = -1
mseq[mseq>0] =  1

# plt.figure()
# plt.plot(mseq)

# a = sp.fft(mseq[:,0])
# f = np.arange(0,fs,fs/a.shape[0])
# plt.figure()
# plt.plot(f,10*np.log10(np.abs(a)**2))


for subject in Subjects:
    
    exclude = ['EXG3','EXG4','EXG5','EXG6','EXG7','EXG8']; #don't need these extra external channels that are saved

    datapath =  os.path.join(data_loc, subject)
    data_eeg,data_evnt = EEGconcatenateFolder(datapath+'/',nchans,refchans,exclude)
    data_eeg.filter(l_freq=2,h_freq=800)
    #data_eeg, data_evnt = data_eeg.resample(4096,events=data_evnt)
    
    if subject == 'E004_Visit_1':
        data_eeg.info['bads'].append('A24')
        data_eeg.info['bads'].append('A7')
    elif subject == 'E004_Visit_2':
        data_eeg.info['bads'].append('A24')
        data_eeg.info['bads'].append('A7')
    elif subject == 'E004_Visit_3':
        data_eeg.info['bads'].append('A24')
        data_eeg.info['bads'].append('A23')
        data_eeg.info['bads'].append('A7')
        data_eeg.info['bads'].append('A9')
    elif subject == 'E002_Visit_1':
        data_eeg.info['bads'].append('A15')
        data_eeg.info['bads'].append('A28') 
    elif subject == 'E005_Visit_2':
        data_eeg.info['bads'].append('A23')
        data_eeg.info['bads'].append('A24')
        data_eeg.info['bads'].append('A25')
        data_eeg.info['bads'].append('A27')
        data_eeg.info['bads'].append('A28')
        data_eeg.info['bads'].append('A21')
        data_eeg.info['bads'].append('A3')
        data_eeg.info['bads'].append('A4')
        data_eeg.info['bads'].append('A6')
        data_eeg.info['bads'].append('A7')
    elif subject == 'E003':
        data_eeg.info['bads'].append('A24')
    elif subject == 'E006_Visit_1':
        data_eeg.info['bads'].append('A12')
        data_eeg.info['bads'].append('A28')
    elif subject == 'E006_Visit_2':
        data_eeg.info['bads'].append('A6')
        data_eeg.info['bads'].append('A28')
    elif subject == 'E007_Visit_1':
        data_eeg.info['bads'].append('A21')
        data_eeg.info['bads'].append('A24')
        data_eeg.info['bads'].append('A15')        
        data_eeg.info['bads'].append('A11')      
        data_eeg.info['bads'].append('A1')  
        data_eeg.info['bads'].append('A7')  
        data_eeg.info['bads'].append('A30')  
    elif subject == 'E007_Visit_3':
        data_eeg.info['bads'].append('A11')
        data_eeg.info['bads'].append('A24')
        data_eeg.info['bads'].append('A7')
    elif subject == 'E012_Visit_1':
        data_eeg.info['bads'].append('A7')
    elif subject == 'E012_Visit_2':
        data_eeg.info['bads'].append('A7')
    elif subject == 'E012_Visit_3':
        data_eeg.info['bads'].append('A20')
    elif subject == 'E014':
        data_eeg.info['bads'].append('A10')
        data_eeg.info['bads'].append('A28')
    elif subject == 'E022_Visit_2':
        data_eeg.info['bads'].append('A21')
    
    #%% Blink Removal
    blinks = find_blinks(data_eeg,ch_name = ['A1'],thresh = 100e-6, l_trans_bandwidth = 0.5, l_freq =1.0)
    blink_epochs = mne.Epochs(data_eeg,blinks,998,tmin=-0.25,tmax=0.25,proj=False,
                                  baseline=(-0.25,0),reject=dict(eeg=500e-6))
    Projs = compute_proj_epochs(blink_epochs,n_grad=0,n_mag=0,n_eeg=8,verbose='DEBUG')
    
    if subject ==  'E001_1':
            ocular_projs = [Projs[0]]#,Projs[1]]
            
    ocular_projs = [Projs[0]]
    
    data_eeg.add_proj(ocular_projs)
    data_eeg.plot_projs_topomap()
    #data_eeg.plot(events=blinks,show_options=True)
    
    del ocular_projs, blink_epochs, Projs, blinks
    
    #%% Plot data and extract epochs
    fs = data_eeg.info['sfreq']
    Reject_Thresh = 200e-6
    if subject[0:4] == 'E004':
        Reject_Thresh = 200e-6
    elif subject == 'E002_Visit_1':
        Reject_Thresh = 250e-6
    elif subject == 'E002_Visit_2':
        Reject_Thresh = 300e-6
    elif subject == 'E002_Visit_2':
        Reject_Thresh = 200e-6
    elif subject == 'E005_Visit_2':
        Reject_Thresh = 200e-6
    reject = dict(eeg=Reject_Thresh)
    epochs = []
    # Evoked_resp = []
    for j in range(2):
        epochs.append(mne.Epochs(data_eeg, data_evnt, [j+1], tmin=-0.5, 
                                  tmax=np.ceil(mseq.size/fs),reject=reject, reject_tmin = 0, reject_tmax=mseq.size/fs, baseline=(-0.2, 0.)) )
        # Evoked_resp.append(epochs[j].average())
        #Evoked_resp[j]
        epochs[j].average().plot(picks=[31],titles=str(j))
    
        info_obj = epochs[0].info
    #%% Extract part of response when stim is on
    ch_picks = np.arange(32)
    remove_chs = []
    if subject == 'E004_Visit_1':
        remove_chs = [6,23]
    elif subject == 'E004_Visit_2':
        remove_chs = [6,23]
    elif subject == 'E004_Visit_3':
        remove_chs = [6,8,22,23]
    elif subject == 'E002_Visit_1':
        remove_chs = [14,27]
        
    ch_picks = np.delete(ch_picks,remove_chs)
    
    ch_picks = 31 #just look at cz for now
    
    epdat = []
    evkdat = []
    fs = epochs[0].info['sfreq']
    t = epochs[0].times
    # fs = Evoked_resp[0].info['sfreq']
    # t = Evoked_resp[0].times
    t1 = np.where(t>=0)[0][0]
    t2 = t1 + mseq.size + int(np.round(0.4*fs))
    t = t[t1:t2]  
    t = np.concatenate((-t[-int(np.round(0.4*fs)):0:-1],t[:-1]))
            
    
    for m in range(len(epochs)):
    # for m in range(len(Evoked_resp)):
        epdat.append(epochs[m].get_data(picks=ch_picks)[:,:,t1:t2].transpose(1,0,2))
        # evkdat.append(Evoked_resp[m].data[ch_picks,t1:t2])
    
    
    del epochs
    #%% Remove epochs with large deflections
    # Reject_Thresh=150e-6
    Tot_trials = np.zeros([len(mseq)])
    for m in range(len(epdat)):
        Peak2Peak = epdat[m].max(axis=2) - epdat[m].min(axis=2)
        mask_trials = np.all(Peak2Peak <Reject_Thresh,axis=0)
        print('rejected ' + str(epdat[m].shape[1] - sum(mask_trials)) + ' trials due to P2P')
        epdat[m] = epdat[m][:,mask_trials,:]
        print('Total Trials Left: ' + str(epdat[m].shape[1]))
        Tot_trials[m] = epdat[m].shape[1]

        
    plt.figure()
    plt.plot(Peak2Peak.T)

    
    #%% Correlation Analysis
        
    Ht = []
    Htnf = []
    # do cross corr
    for m in range(len(epdat)): 
        print('On mseq # ' + str(m+1))
        Ht_m, t_keep = mseqXcorrEpochs_fft(epdat[m],mseq[:,0],fs)
        Ht.append(Ht_m[0,:,:])



    #%% Save Data
    with open(os.path.join(pickle_loc,subject+'_TMTF_cz.pickle'),'wb') as file:
        pickle.dump([t_keep, Ht, info_obj, ch_picks],file)
    del data_eeg, data_evnt, epdat, Ht, info_obj,Htnf, Ht_m
           

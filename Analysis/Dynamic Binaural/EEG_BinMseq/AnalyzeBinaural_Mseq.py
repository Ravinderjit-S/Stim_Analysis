# -*- coding: utf-8 -*-
"""
Created on Thu Sep 06 19:42:24 2018

@author: StuffDeveloping
This analysis is for experiments trying to deconvolve a solution for a varying ITD &IAC
that was varied via an M-sequence 

This code just does filtering, ocular artifact rejection, and decimates the EEG data. 
"""

import matplotlib.pyplot as plt
import os
import pickle
import mne
from anlffr.preproc import find_blinks
from anlffr.preproc import find_saccades
from EEGpp import EEGconcatenateFolder
from mne.preprocessing.ssp import compute_proj_epochs


Subjects = ['S001','S132','S203','S204','S205','S206','S207','S208','S211']
nchans = 34;


#refchans = ['EXG1','EXG2']
refchans = None
    
IAC_eeg = [];
IAC_evnt = [];
ITD_eeg = [];
ITD_evnt = []; 

direct_IAC = '/media/ravinderjit/Data_Drive/Data/EEGdata/DynamicBinaural/IACt/'
direct_ITD = '/media/ravinderjit/Data_Drive/Data/EEGdata/DynamicBinaural/ITDt/'
direct_Mseq = '/media/ravinderjit/Data_Drive/Data/EEGdata/DynamicBinaural/Mseq_4096fs_compensated.mat'

fig_path = os.path.abspath('/media/ravinderjit/Data_Drive/Data/Figures/DynBin/')
pickles_path = os.path.abspath('/media/ravinderjit/Data_Drive/Data/EEGdata/DynamicBinaural/Pickles_32_refAvg')
fig_format = 'png'

exclude = ['EXG1','EXG2','EXG3','EXG4','EXG5','EXG6','EXG7','EXG8']; #don't need these extra external channels that are saved

for s in range(0,len(Subjects)):
    Subject = Subjects[s]
    print('\n\n\n\n' + Subject + '\n\n\n\n')
    IAC_eeg,IAC_evnt = EEGconcatenateFolder(direct_IAC+Subject+'/',nchans,refchans,exclude)
    ITD_eeg,ITD_evnt = EEGconcatenateFolder(direct_ITD+Subject+'/',nchans,refchans,exclude)
    
    IAC_eeg.filter(1,40)
    ITD_eeg.filter(1,40)
    
    ## blink removal
    blinks_IAC = find_blinks(IAC_eeg, ch_name = ['A1'], l_freq=1, l_trans_bandwidth = 0.5) 
    blinks_ITD = find_blinks(ITD_eeg, ch_name = ['A1'], l_freq=1, l_trans_bandwidth = 0.5)
    

    scalings = dict(eeg=40e-6)
    
    blinkIAC_epochs = mne.Epochs(IAC_eeg,blinks_IAC,998,tmin=-0.25,tmax=0.25,proj=False,
                              baseline=(-0.25,0),reject=dict(eeg=500e-6))
    blinkITD_epochs = mne.Epochs(ITD_eeg,blinks_ITD,998,tmin=-0.25,tmax=0.25,proj=False,
                              baseline=(-0.25,0),reject=dict(eeg=500e-6))
    
    
    Projs_IAC = compute_proj_epochs(blinkIAC_epochs, n_grad=0,n_mag=0,n_eeg=5)
    Projs_ITD = compute_proj_epochs(blinkITD_epochs, n_grad=0,n_mag=0,n_eeg=5)
      
    if Subject == 'S211':                  
        eye_projsIAC = [Projs_IAC[0], Projs_IAC[1]] 
        eye_projsITD = [Projs_ITD[0], Projs_IAC[1]] 
    elif Subject == 'S203': 
        eye_projsIAC = [Projs_IAC[0],Projs_ITD[1]] 
        eye_projsITD = [Projs_ITD[0],Projs_ITD[1]] 
    elif Subject == 'S204': 
        eye_projsIAC = [Projs_IAC[0],Projs_ITD[1]]
        eye_projsITD = [Projs_ITD[0],Projs_ITD[1]] 
    elif Subject == 'S132': 
        eye_projsIAC = [Projs_IAC[0],Projs_IAC[1]] 
        eye_projsITD = [Projs_ITD[0],Projs_ITD[1]] 
    elif Subject == 'S206': 
        eye_projsIAC = [Projs_IAC[0],Projs_IAC[1]] 
        eye_projsITD = [Projs_ITD[0],Projs_ITD[1]] 
    elif Subject == 'S207': 
        eye_projsIAC = [Projs_IAC[0]] 
        eye_projsITD = [Projs_ITD[0]]
    elif Subject == 'S205': 
        eye_projsIAC = [Projs_IAC[0]] 
        eye_projsITD = [Projs_ITD[0]] 
    elif Subject == 'S001': 
        eye_projsIAC = [Projs_IAC[0]]
        eye_projsITD = [Projs_ITD[0]]
    elif Subject == 'S208':
        eye_projsIAC = [Projs_IAC[0]] 
        eye_projsITD = [Projs_ITD[0]] 
        
    IAC_eeg.add_proj(eye_projsIAC)
    IAC_eeg.plot_projs_topomap()
    plt.savefig(os.path.join(fig_path,'EEG_ocularProjs/', Subject + '_IAC.' + fig_format),format=fig_format)
    plt.close()
    # IAC_eeg.plot(events=blinks_IAC,scalings=scalings,show_options=True,title = 'IACt')
    ITD_eeg.add_proj(eye_projsITD)
    ITD_eeg.plot_projs_topomap()
    plt.savefig(os.path.join(fig_path,'EEG_ocularProjs/', Subject + '_ITD.' + fig_format),format=fig_format)
    plt.close()
    # ITD_eeg.plot(events=blinks_ITD,scalings=scalings,show_options=True,title ='ITDt')
    
    del eye_projsIAC, eye_projsITD, blinkIAC_epochs, blinkITD_epochs, blinks_ITD, blinks_IAC
    del Projs_IAC, Projs_ITD
    
    channels = [31]
    ylim_vals = [-7,7]
    
    StimIAC_epochs = mne.Epochs(IAC_eeg,IAC_evnt,1,tmin=-0.5,tmax=14,proj=True,baseline=(-0.2, 0.),reject=None)
    StimIAC_epochs.average().plot(picks=channels,titles ='IACt_evoked')
    
    dat = StimIAC_epochs.get_data()
    dat = dat[:,:32,:]
    
    StimITD_epochs = mne.Epochs(ITD_eeg,ITD_evnt,1,tmin=-0.5,tmax=14,proj=True,baseline=(-0.2, 0.),reject=None)
    StimITD_epochs.average().plot(picks=channels,titles ='ITDt_evoked')
    
    del IAC_eeg, ITD_eeg, IAC_evnt, ITD_evnt
    
    StimIAC_epochs.decimate(2) #files are a bit too big so downsampling by factor of 2
    StimITD_epochs.decimate(2)
    StimIAC_epochs.load_data()
    StimITD_epochs.load_data()

    
    with open(os.path.join(pickles_path, Subject + '_DynBin.pickle'),'wb') as f:
        pickle.dump([StimIAC_epochs,StimITD_epochs],f)
        
    del StimIAC_epochs, StimITD_epochs
    
    

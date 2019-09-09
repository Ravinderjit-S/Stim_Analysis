# -*- coding: utf-8 -*-
"""
Created on Thu Sep 06 19:42:24 2018

@author: StuffDeveloping
This analysis is for experiments trying to deconvolve a solution for a varying ITD &IAC
that was varied via an M-sequence 
"""

import numpy as np
import scipy as sp
import pylab as pl
import matplotlib.pyplot as plt
import scipy.io as sio
import mne
from anlffr.preproc import find_blinks
from EEGpp import EEGconcatenateFolder
from mne.preprocessing.ssp import compute_proj_epochs

Subjects = ['S001','S132','S203','S204','S205','S206','S207','S208','S211']
Subject = Subjects[8]

nchans = 34;
#if Subject == 'S203':
#    refchans = ['EXG3', 'EXG4']
#else:    
#    refchans = ['EXG1','EXG2']
refchans = ['EXG1','EXG2']
    
IAC_eeg = [];
IAC_evnt = [];
ITD_eeg = [];
ITD_evnt = []; 

direct_IAC = '/media/ravinderjit/Data_Drive/EEGdata/Binaural/IACt/'
direct_ITD = '/media/ravinderjit/Data_Drive/EEGdata/Binaural/ITDt/'
direct_Mseq = '/media/ravinderjit/Data_Drive/Dropbox/Purdue/Lab/EEGanalysis/Mseq_4096fs_compensated.mat'
#direct_IAC = 'G:/OneDrive - purdue.edu/Purdue/EEGdata/Binaural/IACt/'
#direct_IAC = 'G:\OneDrive - purdue.edu\Purdue\EEGdata\Binaural\IACt\'
#direct_ITD = 'G:\OneDrive - purdue.edu\Purdue\EEGdata\Binaural\ITDt\'

exclude = ['EXG3','EXG4','EXG5','EXG6','EXG7','EXG8']; #don't need these extra external channels that are saved
#if Subject == 'Rav': #accidentally saved a bunch of empty channels so removing them
#    exclude = exclude + ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10', 'B11', 'B12', 'B13', 'B14', 'B15', 'B16', 'B17', 'B18', 'B19', 'B20', 'B21', 'B22', 'B23', 'B24', 'B25', 'B26', 'B27', 'B28', 'B29', 'B30', 'B31', 'B32', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16', 'C17', u'C18', 'C19', 'C20', 'C21', 'C22', 'C23', 'C24', 'C25', 'C26', 'C27', 'C28', 'C29', 'C30', 'C31', 'C32', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'D11', 'D12', 'D13', 'D14', 'D15', 'D16', 'D17', 'D18', 'D19', 'D20', 'D21', 'D22', 'D23', 'D24', 'D25', 'D26', 'D27', 'D28', 'D29', 'D30', 'D31', 'D32']     
#elif Subject == 'S203': #accidentally saved some sensor thing and plugged references into 3&4 instead of 1&2 like usual
#    exclude = exclude[2:] + ['GSR1','GSR2', 'Erg1', 'Erg2', 'Resp','Plet','Temp','EXG1','EXG2'] 


IAC_eeg,IAC_evnt = EEGconcatenateFolder(direct_IAC+Subject+'/',nchans,refchans,exclude)
ITD_eeg,ITD_evnt = EEGconcatenateFolder(direct_ITD+Subject+'/',nchans,refchans,exclude)
IAC_eeg.filter(1,40)
ITD_eeg.filter(1,40)

## blink removal
blinks_IAC = find_blinks(IAC_eeg, ch_name = ['A1'], thresh = 100e-6,  l_trans_bandwidth=0.5, l_freq = 1.0) 
blinks_ITD = find_blinks(ITD_eeg, ch_name = ['A1'], thresh = 100e-6, l_trans_bandwidth=0.5, l_freq =1.0)
scalings = dict(eeg=40e-6)

blinkIAC_epochs = mne.Epochs(IAC_eeg,blinks_IAC,998,tmin=-0.25,tmax=0.25,proj=False,
                          baseline=(-0.25,0),reject=dict(eeg=500e-6))
blinkITD_epochs = mne.Epochs(ITD_eeg,blinks_ITD,998,tmin=-0.25,tmax=0.25,proj=False,
                          baseline=(-0.25,0),reject=dict(eeg=500e-6))
Projs_IAC = compute_proj_epochs(blinkIAC_epochs, n_grad=0,n_mag=0,n_eeg=8,verbose='DEBUG')
Projs_ITD = compute_proj_epochs(blinkITD_epochs, n_grad=0,n_mag=0,n_eeg=8,verbose='DEBUG')
#IAC_eeg.add_proj(Projs_IAC)   
#ITD_eeg.add_proj(Projs_ITD)
#IAC_eeg.plot_projs_topomap()
#ITD_eeg.plot_projs_topomap()
  
if Subject == 'S211':                     
    eye_projsIAC = [Projs_IAC[0],Projs_IAC[2]]
    eye_projsITD = [Projs_ITD[0],Projs_ITD[2]]
elif Subject == 'S203':
    eye_projsIAC = [Projs_IAC[0],Projs_IAC[1]]
    eye_projsITD = [Projs_ITD[0],Projs_ITD[1]]
elif Subject == 'S204':
    eye_projsIAC = [Projs_IAC[0],Projs_IAC[2]]
    eye_projsITD = [Projs_ITD[0],Projs_ITD[2]]
elif Subject == 'S132':
    eye_projsIAC = [Projs_IAC[0],Projs_IAC[2]]
    eye_projsITD = [Projs_ITD[0],Projs_ITD[2]]
elif Subject == 'S206':
    eye_projsIAC = [Projs_IAC[0],Projs_IAC[1]]
    eye_projsITD = [Projs_ITD[0],Projs_ITD[1]]
elif Subject == 'S207':
    eye_projsIAC = [Projs_IAC[0],Projs_IAC[2]]
    eye_projsITD = [Projs_ITD[0],Projs_ITD[2]]
elif Subject == 'S205':
    eye_projsIAC = [Projs_IAC[0],Projs_IAC[2]]
    eye_projsITD = [Projs_ITD[0],Projs_ITD[2]]
elif Subject == 'S001':
    eye_projsIAC = [Projs_IAC[0],Projs_IAC[2]]
    eye_projsITD = [Projs_ITD[0],Projs_ITD[2]]
elif Subject == 'S208':
    eye_projsIAC = [Projs_IAC[0],Projs_IAC[2]]
    eye_projsITD = [Projs_ITD[0],Projs_ITD[2]]
    
IAC_eeg.add_proj(eye_projsIAC)
IAC_eeg.plot_projs_topomap()
IAC_eeg.plot(events=blinks_IAC,scalings=scalings,show_options=True,title = 'IACt')
ITD_eeg.add_proj(eye_projsITD)
ITD_eeg.plot_projs_topomap()
ITD_eeg.plot(events=blinks_ITD,scalings=scalings,show_options=True,title ='ITDt')

channels = [31,4,26,25,30]
ylim_vals = [-7,7]

StimIAC_epochs = mne.Epochs(IAC_eeg,IAC_evnt,1,tmin=-0.5,tmax=14,proj=False)
StimIAC_evoked = StimIAC_epochs.average()
StimIAC_evoked.plot(picks=channels,titles ='IACt_evoked')#ylim = dict(eeg=ylim_vals)) #plotting just to visualize

StimITD_epochs = mne.Epochs(ITD_eeg,ITD_evnt,1,tmin=-0.5,tmax=14,proj=False)
StimITD_evoked = StimITD_epochs.average()
StimITD_evoked.plot(picks=channels,titles ='ITDt_evoked')#ylim=dict(eeg=ylim_vals))

StimIAC_epochs = mne.Epochs(IAC_eeg,IAC_evnt,1,tmin=0,tmax=12.75)
StimIAC_evoked = StimIAC_epochs.average()
StimIAC_evoked.plot(picks=channels,titles ='IACt_evoked')#ylim = dict(eeg=ylim_vals))
    
StimITD_epochs = mne.Epochs(ITD_eeg,ITD_evnt,1,tmin=0,tmax=12.75)
StimITD_evoked = StimITD_epochs.average()
StimITD_evoked.plot(picks=channels,titles ='ITDt_evoked')

Mseq_mat = sio.loadmat(direct_Mseq)
Mseq = Mseq_mat['Mseq_sig'][0,:]
Mseq = Mseq.astype('float')

#plt.figure()
#lag,c,line,b = pl.xcorr(StimIAC_evoked.data[31,:],Mseq,maxlags=4096)
#plt.figure()
#lag,c,line,b = pl.xcorr(StimITD_evoked.data[31,:],Mseq,maxlags=4096)
sio.savemat(str('IAC_evoked20_' +Subject+ '.mat'), {'IAC_EEG_avg':StimIAC_evoked.data})
sio.savemat(str('ITD_evoked20_' +Subject+ '.mat'), {'ITD_EEG_avg':StimITD_evoked.data})
sio.savemat(str('IAC_epochs20_' +Subject+ '.mat'), {'IAC_EEG_epochs':StimIAC_epochs.get_data()[:,0:34,:]})
sio.savemat(str('ITD_epochs20_' +Subject+ '.mat'), {'ITD_EEG_epochs':StimITD_epochs.get_data()[:,0:34,:]})







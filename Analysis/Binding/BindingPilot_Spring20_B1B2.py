#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 11:08:47 2020

@author: ravinderjit
This code will analyze EEG data collected from a binding stimulus in a pilot experiment conducted in Spring, 2020
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
import os
import pickle


fig_format = 'png'
close_plots = True

data_loc = os.path.abspath('/media/ravinderjit/Data_Drive/Data/EEGdata/Binding/BindingPilot_Spring20/B1B2')
dataAnalyzd_loc = os.path.join(data_loc,'Pickles')
nchans = 34;
refchans = ['EXG1','EXG2']

EEG_types = ['Active']
subjects = ['S211_noise']

for m in range(0,len(subjects)):
    subject = subjects[m]
    for k in range(0,len(EEG_types)):
        EEG_type = EEG_types[k]
        print('\n' + '... ' + subject + '  ..........  ' + EEG_type +'\n')
        fig_path = os.path.abspath('/media/ravinderjit/Data_Drive/Data/Figures/BindingPilot/B1B2/'+subject)
        # fig_path = os.path.abspath('/media/ravinderjit/Data_Drive/Data/Figures/BindingPilot/Monaural/'+subject)
        fig_path_blinkprojs = os.path.abspath('/media/ravinderjit/Data_Drive/Data/Figures/BindingPilot/BlinkProjections')
        
        
        if EEG_type =='Active':
             data_path = os.path.join(data_loc, 'Active', subject)
        else:
             data_path = os.path.join(data_loc, 'Passive', subject)    
          
        
        fig_name = subject+'_'+EEG_type
        title_base = subject+ ' ' + EEG_type + ' '
        
        exclude = ['EXG3','EXG4','EXG5','EXG6','EXG7','EXG8']; #don't need these extra external channels that are saved
        
        data_eeg,data_evnt = EEGconcatenateFolder(data_path+'/',nchans,refchans,exclude)
        data_eeg.filter(l_freq=2,h_freq=120)
        
        
        ## blink removal
        blinks_eeg = find_blinks(data_eeg, ch_name = ['A1'], thresh = 100e-6,  l_trans_bandwidth=0.5, l_freq = 1.0) 
        scalings = dict(eeg=40e-6,stim=0.1)
        
        blink_epochs = mne.Epochs(data_eeg,blinks_eeg,998,tmin=-0.25,tmax=0.25,proj=False,
                              baseline=(-0.25,0),reject=dict(eeg=500e-6))
        
        Projs_data = compute_proj_epochs(blink_epochs, n_grad=0,n_mag=0,n_eeg=8,verbose='DEBUG')
        
        # data_eeg.add_proj(Projs_data)   
        # data_eeg.plot_projs_topomap()
        
        # if subject == 'S132':
        eye_projs = Projs_data[0]
        
        
        data_eeg.add_proj(eye_projs)
        data_eeg.plot_projs_topomap()
        plt.savefig(os.path.join(fig_path_blinkprojs,fig_name+'_blinkproj'),format=fig_format)
        if close_plots: plt.close()
        
        # data_eeg.plot(events=blinks_eeg,scalings=scalings,show_options=True)
        
        del Projs_data, blink_epochs, blinks_eeg, eye_projs
        
        # channels = [30,31,26,5,4,27]
        # channels = [25,23]
        channels = [30,31]
        vis_channels = [15,17]
        ylim_vals = [-3.5, 5]
        ts = -0.5
        te = 5
        
        # if subject == 'S227':
        #     ylim_vals = [-4.5, 7]
            
        # if subject == 'S228':
        #     ylim_vals = [-4.5, 7.5]
        
        # epochsAll = mne.Epochs(data_eeg, data_evnt, [1, 2, 3, 4], tmin= ts, tmax= te, proj=True,reject=dict(eeg=200e-6), baseline=(-0.2, 0.),decim=8) 
        epochsAll = mne.Epochs(data_eeg, data_evnt, [1, 2], tmin= ts, tmax= te, proj=True,reject=dict(eeg=200e-6), baseline=(-0.2, 0.),decim=8) 
        evokedAll = epochsAll.average()
        
        epochs_1 = mne.Epochs(data_eeg, data_evnt, [1], tmin= ts, tmax= te, proj=True,reject=dict(eeg=200e-6), baseline=(-0.2, 0.),decim = 8) 
        evoked_1 = epochs_1.average()
        
        epochs_2 = mne.Epochs(data_eeg, data_evnt, [2], tmin= ts, tmax= te, proj=True,reject=dict(eeg=200e-6), baseline=(-0.2, 0.),decim=8) 
        evoked_2 = epochs_2.average()
        
        # epochs_3 = mne.Epochs(data_eeg, data_evnt, [3], tmin= ts, tmax= te, proj=True,reject=dict(eeg=200e-6), baseline=(-0.2, 0.),decim=8) 
        # evoked_3 = epochs_3.average()
        
        # epochs_4 = mne.Epochs(data_eeg, data_evnt, [4], tmin= ts, tmax= te, proj=True,reject=dict(eeg=200e-6), baseline=(-0.2, 0.),decim=8) 
        # evoked_4 = epochs_4.average()
        

        
        evokedAll.plot(picks=channels, titles= title_base +'all events')
        evoked_1.plot(picks=channels,titles=title_base +'Event 1')
        plt.savefig(os.path.join(fig_path,fig_name+'_E1evkd'),format=fig_format)
        if close_plots: plt.close()
        evoked_2.plot(picks=channels,titles=title_base +'Event 2')
        plt.savefig(os.path.join(fig_path,fig_name+'_E2evkd'),format=fig_format)
        if close_plots: plt.close()
        # evoked_3.plot(picks=channels,titles=title_base +'Event 3') 
        # plt.savefig(os.path.join(fig_path,fig_name+'_E3evkd'),format=fig_format)
        # if close_plots: plt.close()
        # evoked_4.plot(picks=channels,titles=title_base +'Event 4') 
        # plt.savefig(os.path.join(fig_path,fig_name+'_E4evkd'),format=fig_format)
        # if close_plots: plt.close()
        
        
        # evokedAll.plot(picks=vis_channels, titles='Visual all events')
        evoked_1.plot(picks=vis_channels,titles='Vis Event 1')
        plt.savefig(os.path.join(fig_path,fig_name+'_E1evkd_vis'),format=fig_format)
        if close_plots: plt.close()
        evoked_2.plot(picks=vis_channels,titles='Vis Event 2')
        plt.savefig(os.path.join(fig_path,fig_name+'_E2evkd_vis'),format=fig_format)
        if close_plots: plt.close()
        # evoked_3.plot(picks=vis_channels,titles='Vis Event 3') 
        # plt.savefig(os.path.join(fig_path,fig_name+'_E3evkd_vis'),format=fig_format)
        # if close_plots: plt.close()
        
        #plot topomaps
        times = np.arange(0.5,4.5,1)
        v_min = -8
        v_max = 8
        
        if subject == 'S227':
            v_min = -3.5
            v_max = 3.5
        
        evoked_1.plot_topomap(times,average=1.0,time_unit='s',proj=True,title= title_base +'E1')#,vmin=v_min,vmax=v_max)
        plt.savefig(os.path.join(fig_path,fig_name+'_E1topo'),format=fig_format)
        if close_plots: plt.close()
        evoked_2.plot_topomap(times,average=1.0,time_unit='s',proj=True,title=title_base +'E2')#,vmin=v_min,vmax=v_max)
        plt.savefig(os.path.join(fig_path,subject+'_'+EEG_type+'_E2topo'),format=fig_format)
        if close_plots: plt.close()
        # evoked_3.plot_topomap(times,average=1.0,time_unit='s',proj=True,title=title_base +'E3')#,vmin=v_min,vmax=v_max)
        # plt.savefig(os.path.join(fig_path,subject+'_'+EEG_type+'_E3topo'),format=fig_format)
        # if close_plots: plt.close()
        
        times2 = np.array([0.1,0.2,0.3,1.1,1.2,1.3,2.1,2.2,2.3,3.1,3.2,3.3,4.1,4.2,4.3])
        evoked_1.plot_topomap(times2,average=0.05,time_unit='s',proj=True,title= title_base +'E1')#,vmin=v_min,vmax=v_max)
        plt.savefig(os.path.join(fig_path,fig_name+'_E1topo_2'),format=fig_format)
        if close_plots: plt.close()
        evoked_2.plot_topomap(times2,average=0.05,time_unit='s',proj=True,title=title_base +'E2')#,vmin=v_min,vmax=v_max)
        plt.savefig(os.path.join(fig_path,subject+'_'+EEG_type+'_E2topo_2'),format=fig_format)
        if close_plots: plt.close()
        # evoked_3.plot_topomap(times2,average=0.05,time_unit='s',proj=True,title=title_base +'E3')#,vmin=v_min,vmax=v_max)
        # plt.savefig(os.path.join(fig_path,subject+'_'+EEG_type+'_E3topo_2'),format=fig_format)
        # if close_plots: plt.close()

        
        
 
        freqs = np.arange(1.,120.,1.)
        T = 1./5
        n_cycles = freqs*T
        time_bandwidth = 2
        vmin = -.1
        vmax = 0.1
        bline = (0,4)
         
        channels = np.arange(0,32)
        
        epochsAll.load_data()
        epochs_1.load_data()
        epochs_2.load_data()
        # epochs_3.load_data()
        # epochs_4.load_data()
        
        tfr_eAll = mne.time_frequency.tfr_multitaper(epochsAll.subtract_evoked(), freqs=freqs, n_cycles = n_cycles, time_bandwidth = time_bandwidth, return_itc = False,picks = channels,decim=1)#,average=False)
        tfr_eAll.plot_topo(baseline =bline,mode= 'logratio', title = 'eAll', vmin=vmin,vmax=vmax)
         
        tfr_e1 = mne.time_frequency.tfr_multitaper(epochs_1.subtract_evoked(), freqs=freqs, n_cycles = n_cycles, time_bandwidth = time_bandwidth, return_itc = False,picks = channels,decim=1)#,average=False)
        tfr_e1.plot_topo(baseline =bline,mode= 'logratio', title = 'e1', vmin=vmin,vmax=vmax)
        
        tfr_e2 = mne.time_frequency.tfr_multitaper(epochs_2.subtract_evoked(), freqs=freqs, n_cycles = n_cycles, time_bandwidth = time_bandwidth, return_itc = False,picks = channels,decim=1)
        tfr_e2.plot_topo(baseline =bline,mode= 'logratio', title = 'e2', vmin=vmin,vmax=vmax)
         
        # tfr_e3 = mne.time_frequency.tfr_multitaper(epochs_3.subtract_evoked(), freqs=freqs, n_cycles = n_cycles, time_bandwidth = time_bandwidth, return_itc = False,picks = channels,decim=1)
        # tfr_e3.plot_topo(baseline =bline,mode= 'logratio', title = 'e3', vmin=vmin,vmax=vmax)
        
        # tfr_e4 = mne.time_frequency.tfr_multitaper(epochs_4.subtract_evoked(), freqs=freqs, n_cycles = n_cycles, time_bandwidth = time_bandwidth, return_itc = False,picks = channels,decim=1)
        # tfr_e4.plot_topo(baseline =bline,mode= 'logratio', title = 'e4', vmin=vmin,vmax=vmax)
         
        vmin = -1
        vmax = 1
        
        # power_e1_evkd = mne.time_frequency.tfr_multitaper(evoked_1, freqs=freqs, n_cycles = n_cycles, time_bandwidth = time_bandwidth, return_itc = False,picks = channels,decim=8)
        # power_e1_evkd.plot_topo(baseline =(-0.2,0),mode= 'logratio', title = 'e1_evkd', vmin=vmin,vmax=vmax)
        
        # power_e2_evkd = mne.time_frequency.tfr_multitaper(evoked_2, freqs=freqs, n_cycles = n_cycles, time_bandwidth = time_bandwidth, return_itc = False,picks = channels,decim=8)
        # power_e2_evkd.plot_topo(baseline =(-0.2,0),mode= 'logratio', title = 'e2_evkd', vmin=vmin,vmax=vmax)
        
        # power_e3_evkd = mne.time_frequency.tfr_multitaper(evoked_3, freqs=freqs, n_cycles = n_cycles, time_bandwidth = time_bandwidth, return_itc = False,picks = channels,decim=8)
        # power_e3_evkd.plot_topo(baseline =(-0.2,0),mode= 'logratio', title = 'e3_evkd', vmin=vmin,vmax=vmax)
        
        #power_e3.plot_joint(timefreqs = (1.4,30) ,picks=[30,31],baseline=(-0.3,0),fmin=28,fmax=40)
 
        with open(os.path.join(dataAnalyzd_loc,subject+'_'+EEG_type+'_tfr.pickle'),'wb') as f:
            #pickle.dump([tfr_e1,tfr_e2,tfr_e3, tfr_e4, tfr_eAll],f)
            pickle.dump([tfr_e1,tfr_e2,tfr_eAll],f)
            
        with open(os.path.join(dataAnalyzd_loc,subject+'_'+EEG_type+'_epochs12.pickle'),'wb') as f:
            pickle.dump([epochs_1,epochs_2],f)
            
        # with open(os.path.join(dataAnalyzd_loc,subject+'_'+EEG_type+'_epochs34.pickle'),'wb') as f:
        #     pickle.dump([epochs_3,epochs_4],f)
            
            
        # del data_eeg, data_evnt
        # del evoked_1,evoked_2,evoked_3
        # del epochs_1,epochs_2,epochs_3
        # del tfr_e1, tfr_e2, tfr_e3
            
            
            
            

        
# picks = [4,25,30,31]
# tfr_evk_data = power_e3_evkd.data[picks,:,:]
# t = power_e3_evkd.times
# b1 = int(np.argwhere(t>=-0.2)[0])
# b2 = int(np.argwhere(t>=0)[0])
# f = power_e3_evkd.freqs
# AvgCh = tfr_evk_data.mean(axis=0)
# AvgCh = 10*np.log10(AvgCh/AvgCh[:,b1:b2].mean(axis=1).reshape(119,1))


# plt.figure()
# vmin = -8
# vmax = 8
# Z = AvgCh
# im = plt.pcolormesh(t,f,Z,vmin=vmin,vmax=vmax)
# axx = im.axes
# plt.colorbar(im)

# def format_coord(x, y):
#     x0, x1 = axx.get_xlim()
#     y0, y1 = axx.get_ylim()
#     col = int(np.floor((x-x0)/float(x1-x0)*t.size))
#     row = int(np.floor((y-y0)/float(y1-y0)*f.size))
#     if col >= 0 and col < Z.shape[1] and row >= 0 and row < Z.shape[0]:
#         z = AvgCh[row, col]
#         return 'x=%1.4f, y=%1.4f, z=%1.4f' % (x, y, z)
#     else:
#         return 'x=%1.4f, y=%1.4f' % (x, y)

# im.axes.format_coord = format_coord
# plt.show()
        

        
        

        
        
        
        
    
    

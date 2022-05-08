#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 21:38:51 2020

@author: ravinderjit
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat 
from scipy.io import savemat 
import psignifit as ps
import os


#%% Load Data

Results_fname = 'MRT_AQ_Rav_results.json'
with open(Results_fname) as f:
    results = json.load(f)
    
    
SNRs = np.arange(7,-6,-3)
subjects = []
accuracy = np.zeros((len(SNRs),len(results)))

pilots = []

#%% Extract Data from Json file

sub_ind = 0
for k in range(0,len(results)):
    if results[k][0]['subject'][:5] == 'PILOT':
        pilots.append(k)
        continue
    
    subjects.append(results[k][0]['subject'])
    
    subj = results[k]
    cur_t = 0
    cond_correct = np.zeros(SNRs.size)
    cond_trials = np.zeros(SNRs.size) 
    for trial in subj:
        if not 'annot' in trial:
            continue
        if len(trial['annot']) == 0:
            continue
    
        cond_t = trial['cond']
        cond_trials[cond_t-1] += 1
        if trial['correct']:
            cond_correct[cond_t-1] += 1
        
        cur_t +=1 
        
    if ~np.all(cond_trials==20):
        print('Something is Fishy!!!!')
    
    accuracy[:,k] = cond_correct / cond_trials
        

accuracy = np.delete(accuracy,pilots,axis=1)

#%% Fit psychometric curves with psignifit

options = dict({
    'sigmoidName': 'norm',
    'expType': 'nAFC',
    'expN': 6
    })

result_ps = []
thresh_70 = np.zeros(len(subjects))
lapse = np.zeros(len(subjects))

psCurves = []

plt.figure()
for sub in range(len(subjects)):
    data_sub = np.concatenate((SNRs[:,np.newaxis], accuracy[:,sub][:,np.newaxis] *cond_trials[:,np.newaxis] , cond_trials[:,np.newaxis] ),axis=1)
    result_sub = ps.psignifit(data_sub,options)
    thresh_70[sub] = ps.getThreshold(result_sub,0.70)[0]
    lapse[sub] = result_sub['Fit'][2]
    
    result_ps.append(result_sub)
    ps.psigniplot.plotPsych(result_sub)
    
    
    #%% Store curves to look at average
    x_vals  = np.linspace(-13, 13, num=1000)
    
    fit = result_sub['Fit']
    data = result_sub['data']
    options = result_sub['options']

    fitValues = (1 - fit[2] - fit[3]) * options['sigmoidHandle'](x_vals,     fit[0], fit[1]) + fit[3]
    
    psCurves.append(fitValues)


#%% Make Box Plot

fig = plt.figure()
fig.set_size_inches(7,8)
plt.rcParams.update({'font.size': 15})
whisker =plt.boxplot(thresh_70)
#whisker['medians'][0].linewidth = 4
plt.xticks([])
plt.yticks([-5,-3, -1])
plt.ylabel('SNR (dB)')

#fig_loc =  '/media/ravinderjit/Data_Drive/Data/Figures/MTBproj/'
#plt.savefig(os.path.join(fig_loc,'MRT_box.svg'),format='svg')




#%% Save data

#savemat('MTB_MRT.mat',{'Subjects':subjects,'thresholds': thresh_70, 'lapse':lapse})




#%%
no_aut = ['5ebd841ddea21b08a0aa3057','5ea492e3b1b3620a6fee3b90','5d86c09df681a1001a8dddc8',
'5e68691662358c10f5adb901','576d8e821a48240001867c54','5b66ce776f0d0400010d678d',
'5d67d4393fcbbc0019f3e1a4','5e8a4368e44c216e867af46b','5d74001d391b6600175f433b',
'5c4ea7c3889752000156ddc5','57950da04a84da00014c5dbb','5ec76194079cdc1b61767c3b',
'5c2805401d4eb70001177bfc','5e72cb561b6e9c21954bd7da','5b78542ef642ed0001517ca0',
'5ddd92855daaa6d095854780','5ea77b0161f86b151f475c8f','5e3606315bfa435ce4d6246c',
'5db37bb9da032c000f9e04fa','5a947fb0f05361000171b5a3','5dd3e3cd5daaa63c5fdb7bb8',
'5c2789fc1694480001e4c4e3','5e2ac2aa0038f21089a533a4','5a791168f49c9a0001f31061',
'5e687fc5651f7d1259eac113','5e6685e1ffed4c405a4db5d0','5eeaff28cf5da71d691ef212',
'5ec37264194de71cb7b82637','5e9b443223881d04be740737','5c024be169406e00014e0b8e',
'5d88db42c1d06e001a1fdcab','5d5f4f294d55db0016ed4612','5c6fb7c3c114eb00018b3154',
'5dda51f040460e9a0f5f6a70','5c5b319722bdd70001afc3ed','5a84f454ae9a0b0001a9e4e5',
'5ef51de92a8ed116654bc910','5e175705cfe8dc000b559793','5b6a87d2cda8590001db8e07',
'5eb81d360943fe74d6aba62b','5ddc516e82a527bc2397e6e1','5ef110ef74978d1e943f0e7a',
'5c01bf7d4c14cf00019ce2a4','5c7c4d0dffaec400019b3699','5eeacbcbb570dc1ab706417a',
'5e7daa886c123242e86c8138','5e9b44743ed5ca041adb1830','56c1adc20aab29000c7d556a',
'5dea9fb3cb53ca1cda9f6116','58f3760092ac81000154f8af',
'5eaf1a2c649fd108284c2b5f','5c08c9f3217d600001117a08','5c5fc0136467ac0001b80940',
'5ed427e3bcd0c00b58a177c0','5e89ce643ac81466c24cb5d8','5d62886927a84f00010fbbb4',
'5f0ec4a78bd5a9220f83e0c2','5e53497a059e37368a22933a','5d100e740277ff00152f7562',
'5dc592c6aa431440aa755d5b','5dc59aea37023940a0860dea','5cb7d2f8f0e73600180b9555',
'5d76d2f7daf4bf00164d585b','5be4ad0aff68b30001975464','5dc47ddd034b45342d3b2e3a',
'5eaefec7c1324e5def56caf8','5e8601d90126b207e8c7a2aa','5d5f07cdca10e90016bc0103',
'5e83adf76ea9870c99d4b086','5ef90b249bafa50a311593ae','5eea4934bde1301131abb0f3',
'5dd6d370194e486498e13ab6','559c3ad8fdf99b32b55f2d32','5dc1c27fb3e5c212018d9e33',
'5dd96ebccf8e7e8fb2261249','5dd43922ecd14c4370d02a2e','5b9eddfed259900001106b05',
'5e500ab1ec45d305b6b82d9f','5a04869ff2e3460001edad2e','5d30f9dfc86fc70001907dfa',
'5f0523936e93408277d06897','5eb2d579ed24d507082b07a2','5d847ac9be4b0b00188463d7',
'5de8695380e0fc7e7ac9f271','5d9f8573ce3daf0015e3e400','5c1c137b0739430001693cf5',
'5f0dbbd17058d70008cc3c99','5cabaa10ed17090015e1eb75','5e4a28f9d897cf49bb42a41c',
'5e820eb36551aa02fd72e1dc','5e7abfb31699ef0bad04e087','5da3dfbcbbbc120019e083ee',
'5c756f32c3c75a0001334269','5d5069303945ac00012780b7','5e2a425609072a0abb76094b',
'5c02ae85b5dd6600019de021','5e32cbad3db17e2868d1f41e','5ad7835e9c198c0001fad31a',
'5ea49733ad3e962d73cc0238','5e9a567a22a6fa0ee4cf44f1','5dffc4b8f9e750be9cd65c07',
'5d42407c2de85600173b5f1e','5caf93416ef0d1001c761380','5ec5ba3e981556631c1a78ba',
'583f02c8ca2e57000184353c',
'5ea8fe56a6d1a135a5a442cd',
'5efc74d0aa189600088be26a',
'5bf461fc707ccc00011cfb75',
'5c732f245858a100013f6f03',
'5e56b9744c97910670a6c6dc',
'5f10e4279df7fd42944120f3',
'5ee0181df2aa853667686d9b',
'5dbdbc9da319ab2ecf2a0887',
'5ebf5b96f621931e939dc6e9',
'5db103ed236ef00013611a6f',
'5eb15e47f5bc2d03f86aa051',
'59f6c60eda08940001d4e8d2',
'5c7ead380ce9a10016fb5ebf',
'5d7499693e40c60001081af3',
'5c1443d21f6f150001494f6a',
'5f107aa051372032695c8e2f',
'5f10a720d6142922737c6038',
'5ef3ba9bfa42131455ca9d80',
'5cbea0b506e27c001e4c03a8',
'5e548644467c4e496a933633',
'5c14dd3f5e545c0001e3f119',
'599e08a45a2c6c0001322a7c',
'5eed22ef1537c7128cca6aaa']


#%% Separate Aut from no Aut

no_aut_here = list(set(no_aut) & set(subjects))
no_aut_ind= []
for sub in no_aut_here:
    no_aut_ind.append(subjects.index(sub))


aut_ind = list(set(no_aut_ind) ^ set(range(len(subjects))))

no_aut_acc = accuracy[:,no_aut_ind]
aut_acc = accuracy[:,aut_ind]

no_aut_sem = no_aut_acc.std(axis=1)/np.sqrt(len(no_aut_ind))
aut_sem = aut_acc.std(axis=1)/np.sqrt(len(aut_ind))

plt.figure()
#plt.plot(SNRs,no_aut_acc.mean(axis=1),label='No Aut')
plt.errorbar(SNRs,no_aut_acc.mean(axis=1),no_aut_sem,label='No Aut')
#plt.plot(SNRs,aut_acc.mean(axis=1),label='Autism')
plt.errorbar(SNRs,aut_acc.mean(axis=1),aut_sem,label='Autism')
plt.legend()

#%% Plot average curves

psCurves = np.array(psCurves)
ps_aut_mean = psCurves[aut_ind,:].mean(axis=0)
ps_nt_mean = psCurves[no_aut_ind,:].mean(axis=0)

ps_aut_sem = psCurves[aut_ind,:].std(axis=0) / np.sqrt(len(aut_ind))
ps_nt_sem = psCurves[no_aut_ind,:].std(axis=0) / np.sqrt(len(no_aut_ind))

ps_mean = psCurves.mean(axis=0)
ps_sem = psCurves.std(axis=0) / np.sqrt(psCurves.shape[0])

fig = plt.figure()
fig.set_size_inches(8,8)
plt.rcParams.update({'font.size': 15})
plt.plot(x_vals,ps_nt_mean, linewidth=2, label='Neurotypical')
plt.fill_between(x_vals, ps_nt_mean - ps_nt_sem, ps_nt_mean + ps_nt_sem, alpha =  0.5)

plt.plot(x_vals,ps_aut_mean, linewidth=2, label= 'Autism')
plt.fill_between(x_vals, ps_aut_mean - ps_aut_sem, ps_aut_mean + ps_aut_sem, alpha =  0.5)

plt.legend()
#plt.xlim([-13,13])
#plt.xticks([-60,-40,-20])
#plt.yticks([0.2,0.6,1])
#plt.ylim([0.1, 1.03])
plt.xlabel('SNR (dB)')
plt.ylabel('Accuracy')

#fig_loc =  '/media/ravinderjit/Data_Drive/Data/Figures/MTBproj/'
#plt.savefig(os.path.join(fig_loc,'MRT_psCurve.svg'),format='svg')

#%% Get AQ scores

data_loc = '/home/ravinderjit/Documents/Data/AQ_prolific/'
AQ = loadmat(data_loc + 'AQscores_Prolific.mat',squeeze_me=True)

aq_subj = AQ['Subjects']
aq_scores = AQ['Scores'].sum(axis=0)

subjs, ind1,ind2 = np.intersect1d(np.array(subjects)[no_aut_ind],aq_subj,return_indices=True)

aq_subj = aq_subj[ind2]
aq_scores = aq_scores[ind2]

plt.figure()
plt.scatter(aq_scores,thresh_70[no_aut_ind]) 


#%% Who hasn't done no_aut

subjs_notDone = np.setdiff1d(np.array(no_aut),np.array(subjects)[no_aut_ind])

#%% Aut pr

subjects_aut = np.array(subjects)[aut_ind]




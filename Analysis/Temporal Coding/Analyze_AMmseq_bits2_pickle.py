
import os
import pickle
import numpy as np
import scipy as sp
import mne
import matplotlib.pyplot as plt
from scipy.signal import freqz
from sklearn.decomposition import PCA


pickle_loc = '/media/ravinderjit/Data_Drive/Data/EEGdata/TemporalCoding/AMmseq_bits2/Pickles/'

m_bits = [7, 10]


with open(os.path.join(pickle_loc,'S211_'+'AMmseqbits2_16384.pickle'),'rb') as file:
    [tdat, Tot_trials, Ht, Htnf,
     info_obj, ch_picks] = pickle.load(file)
    
    
pickle_loc_2 = '/media/ravinderjit/Data_Drive/Data/EEGdata/TemporalCoding/AMmseq_bits4/Pickles_full/'

with open(os.path.join(pickle_loc_2,'S211_'+'AMmseqbits4.pickle'),'rb') as file:
    [tdat_2, Tot_trials_2, Ht_2, Htnf_2,
     info_obj_2, ch_picks_2] = pickle.load(file)
    
    
Keep7_10 = slice(0,6,3)
tdat_2 = tdat_2[Keep7_10]
Tot_trials_2 = Tot_trials_2[Keep7_10]
Ht_2 = Ht_2[Keep7_10]
    
    
#%% Plot time domain 
sbp = [4,4]
sbp2 = [4,4]

fig,axs = plt.subplots(sbp[0],sbp[1],sharex=True,gridspec_kw=None)
for m in range(len(Ht)):
    t = tdat[m]
    Ht_1 = Ht[m]
    for p1 in range(sbp[0]):
        for p2 in range(sbp[1]):
            cur_ch = p1*sbp[1]+p2
            if np.any(cur_ch==ch_picks):
                ch_ind = np.where(cur_ch==ch_picks)[0][0]
                axs[p1,p2].plot(t,Ht_1[ch_ind,:]/np.max(Ht_1[ch_ind,:]))
                axs[p1,p2].set_title(ch_picks[ch_ind])    
                axs[p1,p2].set_xlim([0,0.50])
                # for n in range(int(m*num_nf),int(num_nf*(m+1))):
                #     axs[p1,p2].plot(t,A_Htnf[s][n][ch_ind,:],color='grey',alpha=0.3)
            
fig.suptitle('Ht ')
fig.legend(m_bits)
    
fig,axs = plt.subplots(sbp2[0],sbp2[1],sharex=True,gridspec_kw=None)
for m in range(len(Ht)):    
    t = tdat[m]    
    Ht_1 = Ht[m]
    for p1 in range(sbp2[0]):
        for p2 in range(sbp2[1]):
            cur_ch = p1*sbp2[1]+p2+sbp[0]*sbp[1]
            if np.any(cur_ch==ch_picks):
                ch_ind = np.where(cur_ch==ch_picks)[0][0]
                axs[p1,p2].plot(t,Ht_1[ch_ind,:]/np.max(Ht_1[ch_ind,:]))
                axs[p1,p2].set_title(ch_picks[ch_ind])   
                axs[p1,p2].set_xlim([0,0.50])
                # for n in range(int(m*num_nf),int(num_nf*(m+1))):
                #     axs[p1,p2].plot(t,A_Htnf[s][n][ch_ind,:],color='grey',alpha=0.3)
        
fig.suptitle('Ht ')   
fig.legend(m_bits)
    
    
#%% PCA on tsplit

t_cuts = [.015,.040,.125,.500]

pca_sp_cuts_7 = [list() for i in range(len(t_cuts))]
pca_expVar_cuts_7 = [list() for i in range(len(t_cuts))]
pca_coeff_cuts_7 = [list() for i in range(len(t_cuts))]

pca_sp_cuts_10 = [list() for i in range(len(t_cuts))]
pca_expVar_cuts_10 = [list() for i in range(len(t_cuts))]
pca_coeff_cuts_10 = [list() for i in range(len(t_cuts))]

pca = PCA(n_components=2)

for t_c in range(len(t_cuts)):
    t = tdat[0]
    if t_c ==0:
        t_1 = np.where(t>=0)[0][0]
    else:
        t_1 = np.where(t>=t_cuts[t_c-1])[0][0]
        
    t_2 = np.where(t>=t_cuts[t_c])[0][0]
    
    t_7 = t[t_1:t_2]
    
    pca_sp_cuts_7[t_c] = pca.fit_transform(Ht[0][:,t_1:t_2].T)
    pca_expVar_cuts_7[t_c] = pca.explained_variance_ratio_
    pca_coeff_cuts_7[t_c] = pca.components_
    
    t = tdat[1]
    if t_c ==0:
        t_1 = np.where(t>=0)[0][0]
    else:
        t_1 = np.where(t>=t_cuts[t_c-1])[0][0]
        
    t_2 = np.where(t>=t_cuts[t_c])[0][0]
    
    pca_sp_cuts_10[t_c] = pca.fit_transform(Ht[1][:,t_1:t_2].T)
    pca_expVar_cuts_10[t_c] = pca.explained_variance_ratio_
    pca_coeff_cuts_10[t_c] = pca.components_
    
    t_10 = t[t_1:t_2]
    
    plt.figure()
    plt.title(t_cuts[t_c])
    plt.plot(t_7,pca_sp_cuts_7[t_c][:,1]/np.max(pca_sp_cuts_7[t_c][:,1]))
for m in range(len(Ht)):
    fig,axs = plt.subplots(sbp[0],sbp[1],sharex=True,gridspec_kw=None)
    t = tdat[m]
    Ht_1 = Ht[m]
    t_2 = tdat_2[m]
    Ht_1_2 = Ht_2[m]
    for p1 in range(sbp[0]):
        for p2 in range(sbp[1]):
            cur_ch = p1*sbp[1]+p2
            if np.any(cur_ch==ch_picks):
                ch_ind = np.where(cur_ch==ch_picks)[0][0]
                axs[p1,p2].plot(t,Ht_1[ch_ind,:])
                axs[p1,p2].set_title(ch_picks[ch_ind])    
                axs[p1,p2].set_xlim([0,0.5])
                # for n in range(int(m*num_nf),int(num_nf*(m+1))):
                #     axs[p1,p2].plot(t,A_Htnf[s][n][ch_ind,:],color='grey',alpha=0.3)
            if np.any(cur_ch==ch_picks_2):
                ch_ind = np.where(cur_ch==ch_picks_2)[0][0]
                axs[p1,p2].plot(t_2,Ht_1_2[ch_ind,:])
                axs[p1,p2].set_title(ch_picks[ch_ind])    
                axs[p1,p2].set_xlim([0,0.5])
    fig.suptitle('Ht ' + str(m_bits[m]) + ' bits')
    plt.plot(t_10,pca_sp_cuts_10[t_c][:,1]/np.max(pca_sp_cuts_10[t_c][:,1]))


#%% Compare 16k vs 4k fs
sbp = [4,4]
sbp2 = [4,4]


for m in range(len(Ht)):
    fig,axs = plt.subplots(sbp[0],sbp[1],sharex=True,gridspec_kw=None)
    t = tdat[m]
    Ht_1 = Ht[m]
    t_4k = tdat_2[m]
    Ht_1_2 = Ht_2[m]
    for p1 in range(sbp[0]):
        for p2 in range(sbp[1]):
            cur_ch = p1*sbp[1]+p2
            if np.any(cur_ch==ch_picks):
                t_1 = np.where(t>=0)[0][0]
                t_2 = np.where(t>=0.5)[0][0]
                ch_ind = np.where(cur_ch==ch_picks)[0][0]
                axs[p1,p2].plot(t,Ht_1[ch_ind,:]/np.max(np.abs(Ht_1[ch_ind,t_1:t_2])))
                axs[p1,p2].set_title(ch_picks[ch_ind])    
                axs[p1,p2].set_xlim([0,0.5])
                # for n in range(int(m*num_nf),int(num_nf*(m+1))):
                #     axs[p1,p2].plot(t,A_Htnf[s][n][ch_ind,:],color='grey',alpha=0.3)
            if np.any(cur_ch==ch_picks_2):
                t_1 = np.where(t_4k>=0)[0][0]
                t_2 = np.where(t_4k>=0.5)[0][0]
                ch_ind = np.where(cur_ch==ch_picks_2)[0][0]
                axs[p1,p2].plot(t_4k,Ht_1_2[ch_ind,:]/np.max(np.abs(Ht_1_2[ch_ind,t_1:t_2])))
                axs[p1,p2].set_title(ch_picks_2[ch_ind])    
                axs[p1,p2].set_xlim([0,0.5])
    fig.suptitle('Ht ' + str(m_bits[m]) + ' bits')
    fig.legend(['16k','4k'])
    

for m in range(len(Ht)):   
    fig,axs = plt.subplots(sbp2[0],sbp2[1],sharex=True,gridspec_kw=None)
    t = tdat[m]    
    Ht_1 = Ht[m]
    t_4k = tdat_2[m]
    Ht_1_2 = Ht_2[m]
    for p1 in range(sbp2[0]):
        for p2 in range(sbp2[1]):
            cur_ch = p1*sbp2[1]+p2+sbp[0]*sbp[1]
            if np.any(cur_ch==ch_picks):
                t_1 = np.where(t>=0)[0][0]
                t_2 = np.where(t>=0.5)[0][0]
                ch_ind = np.where(cur_ch==ch_picks)[0][0]
                axs[p1,p2].plot(t,Ht_1[ch_ind,:]/np.max(np.abs(Ht_1[ch_ind,t_1:t_2])))
                axs[p1,p2].set_title(ch_picks[ch_ind])   
                axs[p1,p2].set_xlim([0,0.5])
                # for n in range(int(m*num_nf),int(num_nf*(m+1))):
                #     axs[p1,p2].plot(t,A_Htnf[s][n][ch_ind,:],color='grey',alpha=0.3)
            if np.any(cur_ch==ch_picks_2):
                t_1 = np.where(t_4k>=0)[0][0]
                t_2 = np.where(t_4k>=0.5)[0][0]
                ch_ind = np.where(cur_ch==ch_picks_2)[0][0]
                axs[p1,p2].plot(t_4k,Ht_1_2[ch_ind,:]/np.max(np.abs(Ht_1_2[ch_ind,t_1:t_2])))
                axs[p1,p2].set_title(ch_picks_2[ch_ind])   
                axs[p1,p2].set_xlim([0,0.5])
        
        fig.suptitle('Ht ' + str(m_bits[m]) + ' bits')  
        fig.legend(['16k','4k'])
    
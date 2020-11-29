clear all; close all hidden; clc; %#ok<CLALL>
path = '../CommonExperiment';
p = genpath(path);
addpath(p); %add path to commonly used functions
addpath('Stim_Dev')

subj = input('Please subject ID:', 's');

fig_num=99;
USB_ch=1;
FS_tag = 3;
[f1RZ,PS.RP,FS]=load_play_circuit(FS_tag,fig_num,USB_ch);
%% Stim & Experimental parameters

L=70; %dB SPL
ntrials = 2;
risetime = .050; %taken care of inside stim
TypePhones = 'earphones';
fs =48828.125;
passive =1;

mod4_coh0 = load('CMRrandmod_4_coh_0.mat');
mod4_coh1 = load('CMRrandmod_4_coh_1.mat');
mod40_coh0 = load('CMRrandmod_40_coh_0.mat');
mod40_coh1 = load('CMRrandmod_40_coh_1.mat');
mod223_coh0 = load('CMRrandmod_223_coh_0.mat');
mod223_coh1 = load('CMRrandmod_223_coh_1.mat');
mod_2_10_coh0 = load('CMRrandmod_2_10_coh_0.mat');
mod_2_10_coh1 = load('CMRrandmod_2_10_coh_1.mat');

stim_dur = length(mod4_coh0.Sig(1,:))/fs;

%% Startup parameters
FsamptTDT = 3; %48828.125 Hz
useTrigs =1;
feedback =0;
useTDT = 1;
screenDist = 0.4;
screenWidth = 0.3;
buttonBox =0;

feedbackDuration =0.2;


% Turns EEG Saving on ('Pause off')
invoke(PS.RP, 'SetTagVal', 'trgname',253);
invoke(PS.RP, 'SetTagVal', 'onsetdel',100);
invoke(PS.RP, 'SoftTrg', 6);
pause(2.0);


for i =1:ntrials
    fprintf(1, 'Running Trial #%d/%d\n',i, ntrials*nconds);
    
    stim_mod4_coh0  = [mod4_coh0.Sig(i,:);mod4_coh0.Sig(i,:)];
    stim_mod4_coh1  = [mod4_coh1.Sig(i,:);mod4_coh1.Sig(i,:)];
    stim_mod40_coh0 = [mod40_coh0.Sig(i,:); mod40_coh0.Sig(i,:)];
    stim_mod40_coh1 = [mod40_coh1.Sig(i,:); mod40_coh1.Sig(i,:)];
    stim_mod223_coh0 = [mod223_coh0.Sig(i,:); mod223_coh0.Sig(i,:)];
    stim_mod223_coh1 = [mod223_coh1.Sig(i,:); mod223_coh1.Sig(i,:)];
    stim_mod2_10_coh0 = [mod_2_10_coh0.Sig(i,:); mod_2_10_coh0.Sig(i,:)];
    stim_mod2_10_coh1 = [mod_2_10_coh1.Sig(i,:); mod_2_10_coh1.Sig(i,:)];
    
    PlayStim(stim_mod4_coh0,fs,risetime,PS,L,useTDT, 'NONE', 1, TypePhones);
    WaitSecs(stim_dur + round(stim_dur*0.5));
    
    PlayStim(stim_mod40_coh0,fs,risetime,PS,L,useTDT, 'NONE', 2, TypePhones);
    WaitSecs(stim_dur + round(stim_dur*0.5));
    
    PlayStim(stim_mod223_coh0,fs,risetime,PS,L,useTDT, 'NONE', 3, TypePhones);
    WaitSecs(stim_dur + round(stim_dur*0.5));

    PlayStim(stim_mod4_coh1,fs,risetime,PS,L,useTDT, 'NONE', 4, TypePhones);
    WaitSecs(stim_dur + round(stim_dur*0.5));
    
    PlayStim(stim_mod40_coh1,fs,risetime,PS,L,useTDT, 'NONE', 5, TypePhones);
    WaitSecs(stim_dur + round(stim_dur*0.5));
    
    PlayStim(stim_mod223_coh1,fs,risetime,PS,L,useTDT, 'NONE', 6, TypePhones);
    WaitSecs(stim_dur + round(stim_dur*0.5));
    
    PlayStim(stim_mod2_10_coh0,fs,risetime,PS,L,useTDT, 'NONE', 7, TypePhones);
    WaitSecs(stim_dur + round(stim_dur*0.5));
    
    PlayStim(stim_mod2_10_coh1,fs,risetime,PS,L,useTDT, 'NONE', 8, TypePhones);
    WaitSecs(stim_dur + round(stim_dur*0.5));

end

% % Turns EEG Saving off ('Pause on')
invoke(PS.RP, 'SetTagVal', 'trgname', 254);
invoke(PS.RP, 'SetTagVal', 'onsetdel',100);
invoke(PS.RP, 'SoftTrg', 6);

%Clearing I/O memory buffers:
invoke(PS.RP,'ZeroTag','datainL');
invoke(PS.RP,'ZeroTag','datainR');
pause(3.0);

close_play_circuit(PS.f1,PS.RP);
fprintf(1,'\n Done with data collection!\n');




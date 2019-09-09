

clear all; close all hidden; clc; %#ok<CLALL>
p = genpath('.');
addpath(p);

subj = input('Please subject ID:', 's');

fig_num=99;
USB_ch=1;
FS_tag = 3;
[f1RZ,PS.RP,FS]=load_play_circuit(FS_tag,fig_num,USB_ch);


%% Stim & Experimental parameters

L=70; %dB SPL
ntrials = 1000;
nconds =2;
jitlist = rand(1,ntrials*nconds)*0.1;
risetime = 0.005;
TypePhones = 'earphones';
stim_dur = 1;
fs =48828.125;
passive =1;
BPfilt = designfilt('bandpassfir', 'StopbandFrequency1', 100, 'PassbandFrequency1', 200, 'PassbandFrequency2', 1500, 'StopbandFrequency2', 2000, 'StopbandAttenuation1', 60, 'PassbandRipple', 1, 'StopbandAttenuation2', 60, 'SampleRate', fs);
fmAll = [41 151]; %hz

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

for i =1:ntrials*nconds
    if i <= ntrials
        fm = fmAll(1);
    else
        fm = fmAll(2);
    end
    
    stim = EFR(stim_dur,fs,fm,BPfilt);
    stim = [stim;stim];
    fprintf(1, 'Running Trial #%d/%d\n',i, ntrials*nconds);
    PlayStim_Binaural(stim,fs,risetime,PS,L, useTDT, [], 1, TypePhones,passive)
    WaitSecs(stim_dur + 0.2 + jitlist(i));
end
    
% % Turns EEG Saving off ('Pause on')
invoke(PS.RP, 'SetTagVal', 'trgname', 254);
invoke(PS.RP, 'SetTagVal', 'onsetdel',100);
invoke(PS.RP, 'SoftTrg', 6);





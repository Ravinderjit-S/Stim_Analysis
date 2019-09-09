

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
ntrials = 300;
nconds =4;
jitlist = rand(1,ntrials*nconds)*0.2;
risetime = 0.005;
TypePhones = 'earphones';
stim_dur = 1.5;
fs =48828.125;
passive =1;

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

for i =901:ntrials*nconds
    if i <=300
        stim = load(['stim_mid_lat_ITD' num2str(i) '.mat']); stim=stim.stimITD;
        StimTrigger =1;
    elseif i>= 301 && i<=600
        stim = load(['stim_mid_lat_IPD' num2str(i-300) '.mat']); stim=stim.stimIPD;
        StimTrigger =2;
    elseif i>=601 && i<=900
        stim = load(['stim_lat_lat_ITD2' num2str(i-600) '.mat']); stim=stim.stimITD2;
        StimTrigger =3;
    elseif i>=901
        stim = load(['stim_lat_lat_IPD2' num2str(i-900) '.mat']); stim=stim.stimIPD2;
        StimTrigger =4;
    end
        
    fprintf(1, 'Running Trial #%d/%d\n',i, ntrials*nconds);
    stim = scaleSound(stim);
    PlayStim_Binaural(stim,fs,risetime,PS,L, useTDT, [], StimTrigger, TypePhones,passive)
    WaitSecs(stim_dur + 1.0 + jitlist(i));
end
    
% % Turns EEG Saving off ('Pause on')
invoke(PS.RP, 'SetTagVal', 'trgname', 254);
invoke(PS.RP, 'SetTagVal', 'onsetdel',100);
invoke(PS.RP, 'SoftTrg', 6);





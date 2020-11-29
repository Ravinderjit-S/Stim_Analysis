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

L=75; %dB SPL
ntrials = 300;
nconds =3;
jitlist = rand(1,ntrials*nconds)*0.1;
risetime = .050; %taken care of inside stim
TypePhones = 'earphones';
fs =48828.125;
passive =1;
load('Noise_coherenceABAB_EEGstims.mat')
stim_dur = length(Stim_10(1,:))/fs;

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
    
    fprintf(1, 'Running Trial #%d/%d\n',i, ntrials*nconds);
    stim = [Stim_10(i,:); Stim_10(i,:)];
    PlayStim(stim,fs,risetime,PS,L,useTDT, 'NONE', 1, TypePhones);
    WaitSecs(stim_dur + round(stim_dur*0.5) + jitlist(i));
    
    stim = [Stim_44(i,:);Stim_44(i,:)];
    PlayStim(stim,fs,risetime,PS,L,useTDT, 'NONE', 2, TypePhones);
    WaitSecs(stim_dur + round(stim_dur*0.5) + jitlist(i));
    
    stim = [Stim_109(i,:);Stim_109(i,:)];
    PlayStim(stim,fs,risetime,PS,L,useTDT, 'NONE', 2, TypePhones);
    WaitSecs(stim_dur + round(stim_dur*0.5) + jitlist(i));
     
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




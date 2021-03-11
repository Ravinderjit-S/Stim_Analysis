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
ntrials = 2;
risetime = .050;
TypePhones = 'earphones';
fs =48828.125;
passive =1;

load('mseqEEG_80.mat'); %loads mseqEEG, Point_len

ERB_halfwidth = 0.5;
ERBspacing = 1.5;
target_f = 4000;
noise_bands = CMRbands(target_f, ERB_halfwidth, ERBspacing);

SNRdb = 12;
mod_band = [2 10];

stim_dur = length(mseqEEG)/fs;

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
    
    coh_i = randperm(2)-1;
    
    for m = 1:length(coh_i)
        [Sig] = CMRrandMod_mseqTargetMod(noise_bands,target_f,SNRdb,mod_band,fs,coh_i(m),mseqEEG,Point_len);
        stim = [Sig;Sig];
        PlayStim(stim,fs,risetime,PS,L,useTDT, 'NONE', coh_i(m)+1, TypePhones);
        WaitSecs(stim_dur + 4 + rand*0.05); % wait 4 secs inbetween stims
    end
    
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




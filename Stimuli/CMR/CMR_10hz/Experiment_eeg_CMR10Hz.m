clear all; close all hidden; clc; %#ok<CLALL>
path = '../CommonExperiment';
p = genpath(path);
addpath(p); %add path to commonly used functions
subj = input('Please subject ID:', 's');

fig_num=99;
USB_ch=1;
FS_tag = 3;
[f1RZ,PS.RP,FS]=load_play_circuit(FS_tag,fig_num,USB_ch);
%% Stim & Experimental parameters

L=75; %dB SPL
risetime = .00; %ramping handled in stimulus function
TypePhones = 'earphones';
fs =48828.125;
passive =1;

fc = 4000;
fm = 10;
ofmbw = 1;
ofSNR = 9; % paremterize 
flankdist = 1.5;
flankbw = 1;
condition = [1 2];
stim_dur = 2;
ramp =0.01; %ramping in play stim function
target_modfs = [40, 223];

ntrials = 300;

%% Startup parameters
FsamptTDT = 3; %48828.125 Hz
useTrigs =1;
feedback =0;
useTDT = 1;
screenDist = 0.4;
screenWidth = 0.3;
buttonBox =0;



% Turns EEG Saving on ('Pause off')
invoke(PS.RP, 'SetTagVal', 'trgname',253);
invoke(PS.RP, 'SetTagVal', 'onsetdel',100);
invoke(PS.RP, 'SoftTrg', 6);
pause(2.0);

for i =1:ntrials
    fprintf(1, 'Running Trial #%d/%d\n',i, ntrials);
    tic()
    x_40_2 = makeCMRstim_mod(fc, fs, fm, ofmbw, ofSNR, flankdist, flankbw,...
    condition(2), stim_dur, ramp,target_modfs(1));

    x_40_1 = makeCMRstim_mod(fc, fs, fm, ofmbw, ofSNR, flankdist, flankbw,...
    condition(1), stim_dur, ramp,target_modfs(1));

    x_223_2 = makeCMRstim_mod(fc, fs, fm, ofmbw, ofSNR, flankdist, flankbw,...
    condition(2), stim_dur, ramp,target_modfs(2));

    x_223_1 = makeCMRstim_mod(fc, fs, fm, ofmbw, ofSNR, flankdist, flankbw,...
    condition(1), stim_dur, ramp,target_modfs(2));
    
    stim{1} = [x_40_2; x_40_2];
    stim{2} = [x_40_1;x_40_1];
    stim{3} = [x_223_2;x_223_2];
    stim{4} = [x_223_1; x_223_1];
    stim_gen_t = toc();
    order = randperm(length(stim));
    
    for j =1:length(order)
        if j ==1
            WaitSecs(stim_dur + stim_dur*0.5 - stim_gen_t+0.1*rand());
        end
        PlayStim(stim{order(j)},fs,risetime,PS,L,useTDT, 'NONE', order(j), TypePhones);
        if j ~=length(order) 
            WaitSecs(stim_dur + stim_dur*0.5);
        end
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




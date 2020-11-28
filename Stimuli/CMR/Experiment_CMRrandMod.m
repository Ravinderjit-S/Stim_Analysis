%2 down 1 up experiment in Matlab
clear
clc
path = '../CommonExperiment';
addpath(genpath(path));
addpath('StimDev')

subj = input('Please enter Subject ID:','s');
%% Stim & Experimental Parameters

fs = 44100;
tlen = 1;
t = 0:1/fs:tlen-1/fs;

ERB_halfwidth = 0.5;
ERBspacing = 1.5;
target_f = 4000;
noise_bands = CMRbands(target_f, ERB_halfwidth, ERBspacing);

mod_band = [32 40];
target_modf = 0;
coh = 1;
bp_mod_fo = 1/2 * 5 *fs;

risetime =.050;
TypePhones = 'earphones';

% 2 down, 1 up parameters
Reversals = 0;
Reversals_stop = 11;
Reversals_changeRes = 4;
Start_SNR = 0;
SNR_i = Start_SNR;
StartResolution = 7;
EndResolution = 2;
Correct_2up = [0 0];
TrackSNRs = Start_SNR;
changes =0;

n_mods = load(['RandMod_' num2str(mod_band(2)) '.mat']);
n_mods_iter = 1;


%% Startup parameters
FsampTDT = 3; %48828.125 Hz
useTrigs =1;
feedback =1;
useTDT = 1;
screenDist = 0.4;
screenWidth = 0.3;
buttonBox =1;

feedbackDuration =0.3;

PS = psychStarter(useTDT,screenDist,screenWidth,useTrigs,FsampTDT);

%Clearing I/O memory buffers:
invoke(PS.RP,'ZeroTag','datainL');
invoke(PS.RP,'ZeroTag','datainR');
pause(3.0);

%% Welcome to experiment
textlocH = PS.rect(3)/4;
textlocV = PS.rect(4)/3;
line2line = 50;
ExperimentWelcome(PS, buttonBox,textlocH,textlocV,line2line);

%% Experiment Loop

while Reversals < Reversals_stop
    if coh == 1
        noise_mod = n_mods.noise_mod(n_mods_iter,:);
        n_mods_iter = n_mods_iter+1;
    else
        noise_mod = n_mods.noise_mod(n_mods_iter:n_mods_iter+2,:);
        n_mods_iter = n_mods_iter+3;
    end
    [stim] = CMR_randMod_giveNoiseMod(noise_bands,target_f,SNR_i,mod_band,target_modf,fs,tlen,noise_mod);
    
    if Reversals < Reversals_changeRes
        ChangeSNR = StartResolution;
    else
        ChangeSNR = EndResolution;
    end
    
    PlayOrder = randperm(3);
    stim = stim(PlayOrder);
    
    for j = 1:3
        PlayStim([stim;stim],fs,risetime,PS,L,useTDT,num2str(j),[],TypePhones);
        WaitSecs(tlen+0.3);
    end
    
    answer = find(PlayOrder==3);
    resp = GetResponse_Feedback(PS,feedback,feedbackDuration,buttonBox,answer,textlocH,textlocV,line2line);
  
    Correct = resp == answer;
    Correct_2up = circshift(Correct_2up,1); Correct_2up(1) = Correct;

    fprintf(1, 'SNR =%d, Response =%d, answer =%d, Correct = %d, Reversals = %d, \n',SNR_i, resp, answer,Correct, Reversals);
        
    if Correct 
        if all(Correct_2up)
            SNR_i = SNR_i - ChangeSNR;
            Correct_2up = [0 0];
        end
    else
        SNR_i = SNR_i + ChangeSNR; 
    end
    
    TrackSNRs = [TrackSNRs SNR_i];%#ok
    CurLength = length(changes);
    changes = sign(diff(TrackSNRs)); changes= changes(abs(changes)>0);
    if length(changes)>1 && changes(end) ~=changes(end-1) && length(changes) > CurLength
        Reversals = Reversals + 1;
    end
end
save(['CMR_SigAM_' num2str(mod_band(2)) 'coh_' num2str(coh) '.mat'],'TrackSNRs','target_modf','target_f','noise_bands','coh','mod_band');
    
Screen('DrawText',PS.window,'Experiment is Over!',PS.rect(3)/2-150,PS.rect(4)/2-25,PS.white);
Screen('DrawText',PS.window,'Thank You for Your Participation!',PS.rect(3)/2-150,PS.rect(4)/2+100,PS.white);
Screen('Flip',PS.window);
WaitSecs(5.0);

%Clearing I/O memory buffers:
invoke(PS.RP,'ZeroTag','datainL');
invoke(PS.RP,'ZeroTag','datainR');
pause(3.0);

% Pause On
invoke(PS.RP, 'SetTagVal', 'trgname', 254);
invoke(PS.RP, 'SetTagVal', 'onsetdel',100);
invoke(PS.RP, 'SoftTrg', 6);

close_play_circuit(PS.f1,PS.RP);
fprintf(1,'\n Done with data collection!\n');
sca;

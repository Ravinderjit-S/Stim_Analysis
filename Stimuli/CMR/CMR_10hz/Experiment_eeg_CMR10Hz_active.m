clear all; close all hidden; clc; %#ok<CLALL>
path = '../../CommonExperiment';
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
ofSNR = 15; % paremterize 
flankdist = 2;
flankbw = 1;
condition = [1 2];
stim_dur = 4;
ramp =0.01; %ramping 
target_modfs = [40, 223];

ntrials = 300;

%% Startup parameters
FsamptTDT = 3; %48828.125 Hz
useTrigs =1;
feedback =1;
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
    if mod(i,20) == 0 % optional break every 20 trials
      % Turns EEG Saving off ('Pause on')
        invoke(PS.RP, 'SetTagVal', 'trgname', 254);
        invoke(PS.RP, 'SetTagVal', 'onsetdel',100);
        invoke(PS.RP, 'SoftTrg', 6);
        
        fprintf(1,'Break %d/%d \n',i/20,4)
        info = sprintf('Break %d/%d! You are about to start trial %d out of %d',i/20,4,i,nconds*ntrials);
        info2 = sprintf('Press any button twice to resume');
        Screen('DrawText',PS.window,info,textlocH,textlocV,PS.white);
        Screen('DrawText',PS.window,info2,textlocH,textlocV+100,PS.white);
        Screen('Flip',PS.window);
        if buttonBox  %Subject pushes button twice
            getResponse(PS.RP);
            getResponse(PS.RP);
        else
            getResponseKb; %#ok<UNRCH>
            getResponseKb;
        end
        Screen('Flip',PS.window);
        % Turns EEG Saving on ('Pause off')
        invoke(PS.RP, 'SetTagVal', 'trgname',253);
        invoke(PS.RP, 'SetTagVal', 'onsetdel',100);
        invoke(PS.RP, 'SoftTrg', 6);
        pause(2.0);
    end
    
    fprintf(1, 'Running Trial #%d/%d\n',i, ntrials);
    %tic()
    x_40_2 = makeCMRstim_mod(fc, fs, fm, ofmbw, ofSNR, flankdist, flankbw,...
    condition(2), stim_dur, ramp,target_modfs(1));

    x_40_1 = makeCMRstim_mod(fc, fs, fm, ofmbw, ofSNR, flankdist, flankbw,...
    condition(1), stim_dur, ramp,target_modfs(1));
    
    %stim_gen_t = toc();
    order = randperm(3);
    order = [1 order+1];
    
    for j =1:4
        if order(j) == 1 || order(j) == 4
            trig = 2; %comodulated
            stim = [x_40_1;x_40_1];
        else
            trig = 1; %codeviant 
            stim = [x_40_2;x_40_2];
        end
        PlayStim(stim,fs,risetime,PS,L,useTDT, 'NONE', trig, TypePhones);
        if j ~=length(order) 
            WaitSecs(stim_dur + 1);
        end
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %  Response Frame
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    correct = find(order==4)-1;
    WaitSecs(0.3); %wait until show dot for response 
    resp = GetResponse_Feedback(PS, feedback, feedbackDuration,buttonBox, correct);
    
    respList = [respList, resp]; %#ok<AGROW>
    correctList = [correctList,correct]; %#ok<AGROW>
    
end
save(strcat(subj, '_CMR10HzActive'), 'respList','correctList');


% % Turns EEG Saving off ('Pause on')
invoke(PS.RP, 'SetTagVal', 'trgname', 254);
invoke(PS.RP, 'SetTagVal', 'onsetdel',100);
invoke(PS.RP, 'SoftTrg', 6);

%Clearing I/O memory buffers:
invoke(PS.RP,'ZeroTag','datainL');
invoke(PS.RP,'ZeroTag','datainR');
pause(3.0);

close_play_circuit(f1RZ,PS.RP);
fprintf(1,'\n Done with data collection!\n');




%This code will be for the first part of the comodulation binding
%experiment collecting only behavioral data on 12 conditions
clear all; close all hidden; clc; %#ok<CLALL>
path = '../CommonExperiment';
p = genpath(path);
addpath(p); %add path to commonly used functions

subj = input('Please subject ID:', 's');


%% Stim & Experimental parameters


%load()
rng(s)
L = 70; %dB SPL
respList = []; %vector that will contain the subject's responses 
jitlist = rand(1, ntrials*nconds)*0.2; %small jit to prevent any periodic background noise becoming in phase with desired signal
risetime = 0.050; %made 50 b/c envelope can change at speed of up to 24 Hz which is .041 secs
TypePhones = 'earphones';
stim_dur = 0.7; %This is set by Stim_bind function, need to regenerate stimuli to change this


%% Startup parameters
FsampTDT = 3; % 48828.125 Hz
useTrigs = 1;
feedback = 1;
useTDT = 1;
screenDist = 0.4;
screenWidth = 0.3;
buttonBox = 1;

feedbackDuration = 0.2;

PS = psychStarter(useTDT,screenDist,screenWidth,useTrigs,FsampTDT);

%Clearing I/O memory buffers:
invoke(PS.RP,'ZeroTag','datainL');
invoke(PS.RP,'ZeroTag','datainR');
pause(3.0);



% % Turns EEG Saving off ('Pause on')
% invoke(RZ, 'SetTagVal', 'trgname', 254);
% invoke(RZ, 'SetTagVal', 'onsetdel',100);
% invoke(RZ, 'SoftTrg', 6);



%% Welcome to experiment
textlocH = PS.rect(3)/4;
textlocV = PS.rect(4)/3;
line2line = 50;

ExperimentWelcome(PS, buttonBox,textlocH,textlocV,line2line);

% Turns EEG Saving on ('Pause off')
invoke(PS.RP, 'SetTagVal', 'trgname',253);
invoke(PS.RP, 'SetTagVal', 'onsetdel',100);
invoke(PS.RP, 'SoftTrg', 6);
pause(2.0);

    
%% Iterating through rest of stimuli

for i=1:numel(Corr_indsPlayed) %600 trials
    
    
    if mod(i,80) == 0 % optional break every 80 trials
        % % Turns EEG Saving off ('Pause on')
        invoke(PS.RP, 'SetTagVal', 'trgname', 254);
        invoke(PS.RP, 'SetTagVal', 'onsetdel',100);
        invoke(PS.RP, 'SoftTrg', 6);
        
        info = sprintf('You are about to start trial %d.',i);
        info2 = sprintf('Break %d/7: Take a break or push a button twice to continue',i/80); %7 breaks
        Screen('DrawText',PS.window,info,textlocH,textlocV,PS.white);
        Screen('DrawText',PS.window,info2,textlocH,textlocV+line2line,PS.white);
        Screen('Flip',PS.window);
        if buttonBox  %Subject pushes button twice
            getResponse(PS.RP);
            getResponse(PS.RP);
        else
            getResponseKb; %#ok<UNRCH>
            getResponseKb;
        end
        % Turns EEG Saving on ('Pause off')
        invoke(PS.RP, 'SetTagVal', 'trgname',253);
        invoke(PS.RP, 'SetTagVal', 'onsetdel',100);
        invoke(PS.RP, 'SoftTrg', 6);
        pause(2.0);
    end

    
    fprintf(1, 'Running Trial #%d/%d\n',i, numel(order));
   
    trig_i = Stim_Trigger{i};
    stim_i = stim{i};
    for j = 1:3
        PlayStim(stim_i(j,:),fs,risetime, PS, L, useTDT, num2str(j),trig_i(j),TypePhones);
        WaitSecs(stim_dur + 0.3); %stim is 0.7 seconds long so 1.2 wait time gives 0.3 seconds between each stim
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %  Response Frame
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    resp = GetResponse_Feedback(PS, feedback, feedbackDuration,buttonBox, correct_ans(i),textlocH,textlocV,line2line);
    
    fprintf(1, 'Response = %d, correct =%d \n', resp, correct_ans(i));
    respList = [respList, resp]; %#ok<AGROW>
    WaitSecs(jitlist(i)); %probably unnecessary b/c of variable response time by subjects but adding just in case
end
save(strcat(subj, '_BindingBehEEG'), 'respList');

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


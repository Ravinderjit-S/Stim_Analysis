%This code will do an active EEG experiment where the subject has to
%discern if the two B stimuli in an ABAB setup are different or the same
clear all; close all hidden; clc; %#ok<CLALL>
path = '../CommonExperiment';
p = genpath(path);
addpath(p); %add path to commonly used functions
addpath('Stim_Dev')
subj = input('Please subject ID:', 's');


%% Stim & Experimental parameters
load('s.mat')
rng(s)

L = 75; %dB SPL
respList = []; %vector that will contain the subject's responses 
correctList = [];
risetime = 0.050; %made 50 b/c envelope can change at speed of up to 24 Hz which is .041 secs
TypePhones = 'earphones';
stim_dur = 4.0; %This is set by Stim_bind function, need to regenerate stimuli to change this
f_start = 100;
f_end = 8000;
Tones_num = 16;
fs = 48828;
load('StimActive_AB1AB2noise.mat') %contains stimuli

nconds = length(Corr_inds);
ntrials = 150; %trials per cond

jitlist = rand(1, ntrials*nconds)*0.2; %small jit to prevent any periodic background noise becoming in phase with desired signal


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

%% Welcome to experiment
textlocH = PS.rect(3)/4;
textlocV = PS.rect(4)/3;
line2line = 50;

ExperimentWelcome(PS, buttonBox,textlocH,textlocV,line2line);
Screen('Flip',PS.window);

% Turns EEG Saving on ('Pause off')
invoke(PS.RP, 'SetTagVal', 'trgname',253);
invoke(PS.RP, 'SetTagVal', 'onsetdel',100);
invoke(PS.RP, 'SoftTrg', 6);
pause(2.0);

    
%% Experiment Begins

for i=1:nconds*ntrials
    
    %% Break
    if mod(i,80) == 0 % optional break every 80 trials
        % % Turns EEG Saving off ('Pause on')
        invoke(PS.RP, 'SetTagVal', 'trgname', 254);
        invoke(PS.RP, 'SetTagVal', 'onsetdel',100);
        invoke(PS.RP, 'SoftTrg', 6);
        
        fprintf(1,'Break ----------- \n')
        
        info = sprintf('Break! You are about to start trial %d out of %d',i,nconds*ntrials);
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

    %% Play Stim
    fprintf(1, 'Running Trial #%d/%d\n',i, ntrials*nconds);
   
    stim = [stims{i};stims{i}];
    trig_i = CorrSet(i);
    PlayStim(stim,fs,risetime,PS,L,useTDT, 'NONE', trig_i, TypePhones);
    WaitSecs(stim_dur);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %  Response Frame
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    if CorrSet(i) ==1
        correct =1;
    else
        correct =2;
    end
    
    WaitSecs(0.5); %wait until show dot for response 
    resp = GetResponse_Feedback(PS, feedback, feedbackDuration,buttonBox, correct);
    
    fprintf(1, ['Response = %d, correct = %d, Corr_set= %d  \n'], resp,correct, CorrSet(i));
    respList = [respList, resp]; %#ok<AGROW>
    correctList = [correctList, correct]; %#ok<AGROW>

    WaitSecs(0.8 + jitlist(i)); % jit probably unnecessary b/c of variable response time by subjects but adding just in case
    
end
save(strcat(subj, '_BindingEEG_Act_B1B2noise'), 'respList','Corr_inds','CorrSet');

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


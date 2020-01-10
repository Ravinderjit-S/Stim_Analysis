%This code will be for the first part of the comodulation binding
%experiment collecting only behavioral data on 12 conditions
clear all; close all hidden; clc; %#ok<CLALL>
path = '../CommonExperiment';
p = genpath(path);
addpath(p); %add path to commonly used functions
addpath('Stim_Dev')
subj = input('Please subject ID:', 's');


%% Stim & Experimental parameters
load('s.mat')
rng(s)

L = 70; %dB SPL
respList = []; %vector that will contain the subject's responses 
correctList = [];
risetime = 0.050; %made 50 b/c envelope can change at speed of up to 24 Hz which is .041 secs
TypePhones = 'earphones';
stim_dur = 1.0; %This is set by Stim_bind function, need to regenerate stimuli to change this
f_start = 100;
f_end = 8000;
Tones_num = 16;
fs = 48828;

Corr_inds{1} = 13:16;
Corr_inds{2} = 11:16;
Corr_inds{3} = 9:16;
Corr_inds{4} = [1, 6, 11, 16];
Corr_inds{5} = [1, 4, 7, 10, 13, 16];


nconds = length(Corr_inds);
ntrials = 20; 

CorrSet = repmat(1:nconds,1,ntrials);
CorrSet = CorrSet(randperm(length(CorrSet)));


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

pause(2.0);

    
%% Experiment Begins

[stimA, stimB, stimA2, ~, ~, ~, ~, ~] = Stim_Bind_ABA(Corr_inds{CorrSet(1)},fs,f_start, f_end, Tones_num, []);
stims = vertcat(stimA, stimA2, stimB);

for i=1:nconds*ntrials
    
    %% Break
    if mod(i,80) == 0 % optional break every 80 trials
        
        fprintf(1,'Break ----------- \n')
        
        info = sprintf('Break! This is break %d out of %d',i/80,floor(nconds*ntrials/80));
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
        pause(2.0);
    end

    %% Play Stim
    fprintf(1, 'Running Trial #%d/%d\n',i, ntrials*nconds);
    
    PlayOrder = randperm(3);
    for j = 1:3
        stim = [stims(PlayOrder(j),:); stims(PlayOrder(j),:)];
        PlayStim(stim,fs,risetime,PS,L,useTDT, num2str(j), [], TypePhones);
        tic();
        if j ==3 && i~= ntrials*nconds
            [stimA, stimB, stimA2, ~, ~, ~, ~, ~] = Stim_Bind_ABA(Corr_inds{CorrSet(i+1)},fs,f_start, f_end, Tones_num, []);
            stims = vertcat(stimA, stimA2, stimB);
            StimGenTime = toc();
        else
            StimGenTime = toc();
        end
        WaitSecs(stim_dur + 0.3 - StimGenTime); %wait 0.3 seconds b/t each stim
    end
    correctList = [correctList, find(PlayOrder==3)];%#ok
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %  Response Frame
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    resp = GetResponse_Feedback(PS, feedback, feedbackDuration,buttonBox, correctList(end));
    
    fprintf(1, ['Response = %d, correct =%d, Corr_inds= [' repmat('%d, ',1,numel(Corr_inds{CorrSet(i)})-1) '%d]\n'], resp,correct, Corr_inds{CorrSet(i)});
    respList = [respList, resp]; %#ok<AGROW>

    WaitSecs(0.3); % jit probably unnecessary b/c of variable response time by subjects but adding just in case
    
end
save(strcat(subj, '_BindingBehavior'), 'respList','correctList','Corr_inds','CorrSet');

Screen('DrawText',PS.window,'Experiment is Over!',PS.rect(3)/2-150,PS.rect(4)/2-25,PS.white);
Screen('DrawText',PS.window,'Thank You for Your Participation!',PS.rect(3)/2-150,PS.rect(4)/2+100,PS.white);
Screen('Flip',PS.window);
WaitSecs(5.0);

%Clearing I/O memory buffers:
invoke(PS.RP,'ZeroTag','datainL');
invoke(PS.RP,'ZeroTag','datainR');
pause(3.0);

close_play_circuit(PS.f1,PS.RP);
fprintf(1,'\n Done with data collection!\n');
sca;


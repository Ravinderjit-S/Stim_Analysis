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

L = 70; %dB SPL
respList = []; %vector that will contain the subject's responses 
risetime = 0.050; %made 50 b/c envelope can change at speed of up to 24 Hz which is .041 secs
TypePhones = 'earphones';
stim_dur = 4.0; %This is set by Stim_bind function, need to regenerate stimuli to change this
f_start = 100;
f_end = 8000;
Tones_num = 16;
fs = 48828;



Corr_inds{1,1} = 5:16; %B1
Corr_inds{1,2} = 1:16; %B2
Corr_inds{2,1} = 5:16;
Corr_inds{2,2} = 5:16;
Corr_inds{3,1} = 1:16;
Corr_inds{3,2} = 5:16;
Corr_inds{4,1} = 1:16;
Corr_inds{4,2} = 1:16;


nconds = length(Corr_inds);
ntrials = 5; %trials per cond

CorrSet = repmat(1:nconds,1,ntrials);
CorrSet = CorrSet(randperm(length(CorrSet)));

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

%% Generate first stimulus
stim = Stim_Bind_AB1AB2(Corr_inds{1,1}, Corr_inds{1,2},fs,f_start, f_end, Tones_num, []);
stim = [stim;stim];


%% Welcome to experiment
textlocH = PS.rect(3)/4;
textlocV = PS.rect(4)/3;
line2line = 50;

ExperimentWelcome(PS, buttonBox,textlocH,textlocV,line2line);
Screen('Flip',PS.window);

%% Demo, just listen

demo1 = true; 
while demo1
    info = sprintf('you will hear A_B1_A_B2');
    info2 = sprintf('Press any button twice to play stim');
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
    PlayStim(stim,fs,risetime,PS,L,useTDT, 'NONE', [], TypePhones);
    tic()
    stim = Stim_Bind_AB1AB2(Corr_inds{1,1}, Corr_inds{1,2},fs,f_start, f_end, Tones_num, []);
    stim = [stim;stim];
    stimGenT =toc();
    
    WaitSecs(stim_dur-stimGenT);
    
    info = sprintf('To hear again, press 1. To continue, press 2');
    Screen('DrawText',PS.window,info,textlocH,textlocV,PS.white);
    Screen('Flip',PS.window);
    resp = getResponse(PS.RP);
    if resp ~= 1
        demo1 = false;
    end
end


Screen('Flip',PS.window);
demo2 = true;
stim = Stim_Bind_AB1AB2(Corr_inds{2,1}, Corr_inds{2,2},fs,f_start, f_end, Tones_num, []);
stim = [stim;stim];

while demo2
    info = sprintf('you will hear A_B1_A_B1');
    info2 = sprintf('Press any button twice to play stim');
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
    PlayStim(stim,fs,risetime,PS,L,useTDT, 'NONE', [], TypePhones);
    tic()
    stim = Stim_Bind_AB1AB2(Corr_inds{2,1}, Corr_inds{2,2},fs,f_start, f_end, Tones_num, []);
    stim = [stim;stim];
    stimGenT =toc();
    
    WaitSecs(stim_dur-stimGenT);
    
    info = sprintf('To hear again, press 1. To continue, press 2');
    Screen('DrawText',PS.window,info,textlocH,textlocV,PS.white);
    Screen('Flip',PS.window);
    resp = getResponse(PS.RP);
    if resp ~= 1
        demo2 = false;
    end
end


Screen('Flip',PS.window);
demo3 = true;
stim = Stim_Bind_AB1AB2(Corr_inds{3,1}, Corr_inds{3,2},fs,f_start, f_end, Tones_num, []);
stim = [stim;stim];

while demo3
    info = sprintf('you will hear A_B2_A_B1');
    info2 = sprintf('Press any button twice to play stim');
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
    PlayStim(stim,fs,risetime,PS,L,useTDT, 'NONE', [], TypePhones);
    tic()
    stim = Stim_Bind_AB1AB2(Corr_inds{3,1}, Corr_inds{3,2},fs,f_start, f_end, Tones_num, []);
    stim = [stim;stim];
    stimGenT =toc();
    
    WaitSecs(stim_dur-stimGenT);
    
    info = sprintf('To hear again, press 1. To continue, press 2');
    Screen('DrawText',PS.window,info,textlocH,textlocV,PS.white);
    Screen('Flip',PS.window);
    resp = getResponse(PS.RP);
    if resp ~= 1
        demo3 = false;
    end
end

Screen('Flip',PS.window);
demo4 = true;
stim = Stim_Bind_AB1AB2(Corr_inds{4,1}, Corr_inds{4,2},fs,f_start, f_end, Tones_num, []);
stim = [stim;stim];

while demo4
    info = sprintf('you will hear A_B2_A_B2');
    info2 = sprintf('Press any button twice to play stim');
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
    PlayStim(stim,fs,risetime,PS,L,useTDT, 'NONE', [], TypePhones);
    tic()
    stim = Stim_Bind_AB1AB2(Corr_inds{4,1}, Corr_inds{4,2},fs,f_start, f_end, Tones_num, []);
    stim = [stim;stim];
    stimGenT =toc();
    
    WaitSecs(stim_dur-stimGenT);
    
    info = sprintf('To hear again, press 1. To continue, press 2');
    Screen('DrawText',PS.window,info,textlocH,textlocV,PS.white);
    Screen('Flip',PS.window);
    resp = getResponse(PS.RP);
    if resp ~= 1
        demo4 = false;
    end
end

info = sprintf('Now for a full practice run!');
info2 = sprintf('Press any button twice when ready');
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
    
%% Experiment Begins

stim = Stim_Bind_AB1AB2(Corr_inds{CorrSet(1),1}, Corr_inds{CorrSet(1),2},fs,f_start, f_end, Tones_num, []);
stim = [stim;stim];

for i=1:nconds*ntrials
    %% Play Stim
    fprintf(1, 'Running Trial #%d/%d\n',i, ntrials*nconds);
   
    
    trig_i = CorrSet(i);
    PlayStim(stim,fs,risetime,PS,L,useTDT, 'NONE', trig_i, TypePhones);
    
    stimGenT = 0;
    if i~=nconds*ntrials
       tic()
       stim = Stim_Bind_AB1AB2(Corr_inds{CorrSet(i+1),1}, Corr_inds{CorrSet(i+1),2},fs,f_start, f_end, Tones_num, []);
       stim = [stim;stim];
       stimGenT =toc();
    end
    WaitSecs(stim_dur - stimGenT);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %  Response Frame
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    if CorrSet(i) ==2 || CorrSet(i) ==4
        correct =1;
    else
        correct =2;
    end
    
    WaitSecs(0.5); %wait until show dot for response 
    resp = GetResponse_Feedback(PS, feedback, feedbackDuration,buttonBox, correct);
    
    fprintf(1, ['Response = %d, correct = %d, Corr_set= %d  \n'], resp,correct, CorrSet(i));
    respList = [respList, resp]; %#ok<AGROW>

    WaitSecs(0.5 + jitlist(i)); % jit probably unnecessary b/c of variable response time by subjects but adding just in case
    
end

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


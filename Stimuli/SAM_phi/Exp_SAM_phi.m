%3 AFC to find highest detectable OSCOR fm

clear all; close all hidden; clc; %#ok<CLALL>
path = '../CommonExperiment';
p = genpath(path);
addpath(p);

load('s.mat')
rng(s)

subj = input('Please subject ID:', 's');
%% Stim & Experimental parameters
AMs_test = [4,64];
phi_test = [15, 30, 45, 60, 75, 90, 180];
L=70; %dB SPL
ntrials = 20;
nconds = numel(AMs_test) * numel(phi_test);

AMs = repmat(AMs_test,1,ntrials*length(phi_test));
phis = repmat(phi_test,1,ntrials * length(AMs_test));

rand_order = randperm(length(AMs));
AMs = AMs(rand_order);
phis = phis(rand_order);

risetime = 0.100;
TypePhones = 'earphones';
stim_dur = 1.0; %duration of each SAM, 3 will be played per trial
fs =48828.125;
respList = [];
correctList = [];

%% Startup parameters
FsampTDT = 3; %48828.125 Hz
useTrigs =1;
feedback =1;
useTDT = 1;
screenDist = 0.4;
screenWidth = 0.3;
buttonBox =1;

feedbackDuration =0.2;

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

f1 = randi(1800) + 200; 
f2 = 4*f1; 
stim = SAM_phi(f1,f2,fs,stim_dur,AMs(1),phis(1)); %first stim

for i =1:ntrials*nconds
    
    if i == round(ntrials*nconds/2) % Break at half way point
        info = strcat('Break! Half way Done: Press any button twice to begin...');
        
        Screen('DrawText',PS.window,info,textlocH,textlocV+line2line,PS.white);
        Screen('Flip',PS.window);
        fprintf(1, 'Break \n');
        if buttonBox  %Subject pushes button twice to begin
            getResponse(PS.RP);
            getResponse(PS.RP);
        else
            getResponseKb; %#ok<UNRCH>
            getResponseKb;
        end
        fprintf(1,'Subject continued \n')
    end
        
    fprintf(1, 'Running Trial #%d/%d\n',i, ntrials*nconds);
    
    PlayOrder= randperm(3);
    stim = stim(PlayOrder);
    for j = 1:3
        PlayStim([stim{j};stim{j}],fs,risetime,PS,L, useTDT, num2str(j), [], TypePhones);
        tic();
        if j == 3 && i~= ntrials*nconds
            f1 = randi(1800) + 200; 
            f2 = 4*f1; 
            stim = SAM_phi(f1,f2,fs,stim_dur,AMs(i+1),phis(i+1)); %first stim
            StimGenTime = toc();
        else
            StimGenTime = toc();
        end
        WaitSecs(stim_dur + 0.3 - StimGenTime); %wait 0.3 seconds b/t each stim
    end
    correctList = [correctList, find(PlayOrder ==3)]; %#ok
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %  Response Frame
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    resp = GetResponse_Feedback(PS, feedback, feedbackDuration,buttonBox, correctList(end));
    
    fprintf(1, 'Response =%d, answer =%d, Correct = %d, fm = %d, phi = %d \n', resp, correctList(end),resp==correctList(end), AMs(i), phis(i));
    respList = [respList, resp]; %#ok<AGROW>
    
end

save([subj '_SAM_phi.mat'],'AMs','ntrials','respList','correctList','phis')

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
    





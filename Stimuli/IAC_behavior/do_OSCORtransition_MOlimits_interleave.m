% Employ method of limits to determine where the OSCOR stimulus goes from
% spatial to a flutter

clear all; close all hidden; clc; %#ok<CLALL>
path = '../CommonExperiment';
p = genpath(path);
addpath(p);
addpath('Stimuli Dev')
load('s.mat')
rng(s)

subj = input('Please subject ID:', 's');


%% Stim & Experimental parameters
L=70; %dB SPL
series_n = 1;
startF = [2 20]; %starting frequencies for ascending and descending


risetime = 0.050;
TypePhones = 'earphones';
stim_dur = 1; 
fs =48828.125;
passive =0;
BPfilt = designfilt('bandpassfir', 'StopbandFrequency1', 100, 'PassbandFrequency1', 200, 'PassbandFrequency2', 1500, 'StopbandFrequency2', 2000, 'StopbandAttenuation1', 60, 'PassbandRipple', 1, 'StopbandAttenuation2', 60, 'SampleRate', fs);


%% Startup parameters
FsampTDT = 3; %48828.125 Hz
useTrigs =1;
feedback =0;
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

for i=1:length(series_n)
    OSCOR_fm_A = startF(1) + randi(4);
    OSCOR_fm_D = startF(2) - randi(4);
    series_opt = ['A' 'D'];
    respList = [];
    fplayed = [];
    splayed = []; %the series for that run ... either Ascending or descending
    respchange = [0 0]; %initializing for while loop ... this is looking at if resp has change for Ascending or Descending trials
    
    info = sprintf('Press any button twice to start block %d/%d',i,length(series_n));
    Screen('DrawText',PS.window,info,textlocH,textlocV,PS.white);
    Screen('Flip',PS.window);
    if buttonBox  %Subject pushes button twice to begin
        getResponse(PS.RP);
        getResponse(PS.RP);
    end
    Screen('Flip',PS.window);
    
    if series_opt(randi(length(series_opt))) == series_opt(1)
        stim = OSCOR(stim_dur,fs,OSCOR_fm_A,BPfilt,0);
        series_cur = series_opt(1);
    else
        stim = OSCOR(stim_dur,fs,OSCOR_fm_D,BPfilt,0);
        series_cur = series_opt(2);
    end
    
    while any(respchange == 0)
        PlayStim(stim,fs,risetime,PS,L,useTDT,'NONE',[], TypePhones);
        WaitSecs(stim_dur + 0.2);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %  Response Frame
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        resp = GetResponse_Feedback(PS, feedback, feedbackDuration,buttonBox, []);

        if series_cur == series_opt(1)
            OSCOR_fm = OSCOR_fm_A;
        else
            OSCOR_fm = OSCOR_fm_D;
        end
        fprintf(1, 'Response =%d, OSCORfm = %d, Series = %s \n', resp, OSCOR_fm,series_cur);
        respList = [respList, resp]; %#ok<AGROW>
        fplayed = [fplayed, OSCOR_fm]; %#ok<AGROW>
        splayed = [splayed, series_cur]; %#ok<AGROW>
        if series_cur == series_opt(1)
            OSCOR_fm_A = OSCOR_fm_A +1;
            if resp == 2 
                respchange(1) = 1; 
            end
        else
            OSCOR_fm_D = OSCOR_fm_D -1;
            if resp ==1
                respchange(2) =1;
            end
        end
        
        if series_opt(randi(length(series_opt))) == 'A' && respchange(1) == 0 || respchange(2) == 1
            stim = OSCOR(stim_dur,fs,OSCOR_fm_A,BPfilt,0);
            series_cur = series_opt(1);
        else
            stim = OSCOR(stim_dur,fs,OSCOR_fm_D,BPfilt,0);
            series_cur = series_opt(2);
        end
    end
    A_respList{i} = respList;
    A_fplayed{i} = fplayed;
    A_splayed{i} = splayed;
end

save([subj '_OSCORtransition_MOLinterleave.mat'], 'A_respList', 'A_fplayed','A_splayed')

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
    





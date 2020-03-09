%3AFC to determine phase difference detectable at different AM rates

clear all; close all hidden; clc; %#ok<CLALL>
path = '../CommonExperiment';
p = genpath(path);
addpath(p);


subj = input('Please subject ID:', 's');
file_load = input('File name for last block to load, type NONE if starting from 1st block:','s');
%% Stim & Experimental parameters
Mods_test = [4,8,16,32,64];
phi_test = [15, 30, 45, 60, 75, 90, 180];

L=70; %dB SPL
ntrials = 20;
nconds = numel(Mods_test) * numel(phi_test);
diotic = 0; %if 1, send carriers to different ears
frange = [500 6000]; % range of the carriers
fratio = 4; % ratio of 2 carriers ... 4 = 2 octaves

AMs = repmat(Mods_test,1,ntrials*length(phi_test));
AM_phis = repmat(phi_test,1,ntrials * length(Mods_test));
FMs = repmat(Mods_test,1,ntrials*length(phi_test));
FM_phis = repmat(phi_test,1,ntrials * length(Mods_test));

rand_order1 = randperm(length(AMs));
AMs = AMs(rand_order1);
AM_phis = AM_phis(rand_order1);
rand_order2 = randperm(length(FMs));
FMs = FMs(rand_order2);
FM_phis = FM_phis(rand_order2);

blockSize = 100;
blocks = ceil(length(AMs)/blockSize) + ceil(length(FMs)/blocksize);

risetime = .125;
TypePhones = 'earphones';
stim_dur = 1.0; %duration of each SAM, 3 will be played per trial
fs =48828.125;
respList = [];
correctList = [];

%% make parameter structure
params.modType = [];
params.mod = [];
params.phi = [];

blockSize_p = blockSize;
for b = 1:blocks
   if (b == blocks || b == blocks-1) && mod(length(AMs),blockSize) ~=0
        blockSize_p = mod(length(AMs) ,blockSize);
   end
   if mod(b,2) == 1 %Do AM block, followed by FM block and go back and forth
       indexes =  ((b+1)/2 - 1)*blockSize_p + 1 : (b+1)/2 *blockSize_p;
       params.modType = [params.modType, 'A'];
       params.mod = [params.mod AMs(indexes)];
       params.phi = [params.phi AM_phis(indexes)];
   else
       indexes = (b/2 -1) * blockSize_p + 1: b/2* blockSize_p;
       params.modType = [params.modType, 'F'];
       params.mod = [params.mod FMs(indexes)];
       params.phi = [params.phi FM_phis(indexes)];
   end
end


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

blockSize_i = blockSize;
trialgen = 1;

start_block=1;
if ~strcmpi(file_load,'NONE')
    load(file_load);
    start_block=str2num(file_load(end-4))+1;
end
for b=start_block:blocks
    if (b == blocks || b == blocks-1) && mod(length(AMs),blockSize) ~=0
        blockSize_i = mod(length(AMs) ,blockSize);
    end
    %% gen first stim of block
    f1 = randi(frange(2)/fratio - frange(1)) + frange(1); 
    f2 = fratio*f1;  
    if params.modType(b) == 'A'
        stim = SAM_phi(f1,f2,fs,stim_dur,params.mod(trialgen),params.phi(trialgen),diotic);
    else
        stim = FM_phi(f1,f2,fs,stim_dur,params.mod(trialgen),params.phi(trialgen),diotic);
    end

    for i =1:blockSize_i 
        %% begin block
        fprintf(1, 'Running Trial #%d/%d in block %d\n',i, blockSize_i, b);
        trialgen = trialgen+1;
        
        PlayOrder= randperm(3);
        stim = stim(PlayOrder);
        for j = 1:3
            PlayStim(stim{j},fs,risetime,PS,L, useTDT, num2str(j), [], TypePhones);
            tic();
            if j == 3 && i~= blockSize
                f1 = randi(frange(2)/fratio - frange(1)) + frange(1); 
                f2 = fratio*f1; 
                if params.modType(b) == 'A'
                    stim = SAM_phi(f1,f2,fs,stim_dur,params.mod(trialgen),params.phi(trialgen),diotic); 
                else
                    stim = FM_phi(f1,f2,fs,stim_dur,params.mod(trialgen),params.phi(trialgen),diotic);
                end
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

        fprintf(1, 'Response =%d, answer =%d, Correct = %d, trial = %d /100 \n',resp, correctList(end),resp==correctList(end), i);
        respList = [respList, resp]; %#ok<AGROW>

    end
    block_acc = sum(respList(end-blockSize_i:end)==correctList(end-blockSize_i:end)) / blockSize_i;
    fprintf(1, 'Block Accuracy: %d %% \n', round(block_acc));
    
    save([subj '_SamFm_phi_block' num2str], 'params', 'ntrials','respList','correctList','trialgen','diotic') 
    if b ~=blocks
        info = sprintf('Break! About to start Block %d/%d: Press any button twice to begin...',b+1,blocks);
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
    
end

save([subj '_SamFm_phi_aBlocks.mat'],'params', 'ntrials','respList','correctList','diotic')

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
    





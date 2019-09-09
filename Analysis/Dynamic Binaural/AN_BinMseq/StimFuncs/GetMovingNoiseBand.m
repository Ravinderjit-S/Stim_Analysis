function [Lsamps,Rsamps,ITDt,dur_real] = GetMovingNoiseBand(params)

global ChanConfigInfo %This global variable is set by the user at startup, and determines which channel is considered "contra-lateral".... taking into account the recording side, and the connections between channels and ears (i.e., this should be robust to someone mistakenly putting the right speaker in the left ear).
if isempty(ChanConfigInfo)
    ChanConfigInfo.ContraChan = 1;
end

%Gather all the required parameters  

RandSeed = params.Inloop.Noise_Seed;
Start_ITD = params.Inloop.Start_ITD;
Binaural_Speed = params.Inloop.Binaural_Speed;
Rho = params.Inloop.Interaural_Correlation;
fLow = params.Inloop.Low_Cut_Freq*1000;
fHigh = params.Inloop.High_Cut_Freq*1000;
% nChan = 2;
dur = params.Gating.Duration;
gateDur = params.Gating.Rise_fall_time;
fm = params.Inloop.ITD_Modulation_Frequency;
MOVNType = params.Inloop.ITD_Modulation_Type;
mseq_ITDstart = params.Inloop.MSeq_ITD_start;
mseq_ITDend = params.Inloop.MSeq_ITD_end;
N = params.Inloop.MSeq_N;
Mseq_dur = params.Inloop.MSeq_StepDuration;
%-----------------------------------
Fsample = 48828.125;%Please don't edit this.
%-----------------------------------


%Some error checking on the inputs
[err] = local_error_check;
if err
    error('problem with your choice of parameters (It''s not me, it''s you!)');
end

switch MOVNType
    case 'Linear'
        %Pass the parameters to the moving noise function
        [movingNoise,ITDt] = MakeMovingNoiseStim(fLow,fHigh,dur,Fsample,RandSeed,Binaural_Speed,Start_ITD*1000,gateDur,Rho);
        dur_real = dur;
    case 'M-Sequence'
        [movingNoise,ITDt,dur_real] = MakeMSeqMovingNoiseStim(dur,Fsample,RandSeed,mseq_ITDstart,mseq_ITDend,Rho,Mseq_dur,N,ChanConfigInfo.ContraChan);
    otherwise
        [movingNoise,ITDt] = MakeOscillatingNoiseStim(MOVNType,fLow,fHigh,dur,Fsample,RandSeed,fm,Start_ITD,gateDur,Rho);
        dur_real = dur;
end
Lsamps{1} = movingNoise(:,1);
Rsamps{1} = movingNoise(:,2);

if strcmp(params.Inloop.Interleave_Channels,'Yes')
    for i =1:length(Lsamps)
        tempL{(i-1)+1} = Lsamps{i};
        tempL{(i-1)+2} = Rsamps{i};
        tempR{(i-1)+1} = Rsamps{i};
        tempR{(i-1)+2} = Lsamps{i};
    end
    Lsamps = tempL;
    Rsamps = tempR;
    clear tempL tempR;
end


    function [err] = local_error_check
        err = 0;
        if length(RandSeed)>1
            nelerror('You can only specify 1 Noise Seed at a time for this template');
            err = 1;
        end
        if length(Binaural_Speed)>1
            nelerror('You can only specify 1 Binaural Speed at a time for this template');
            err = 1;
        end
        if length(Rho)>1
            nelerror('You can only specify 1 value of IAC at a time for this template');
            err = 1;
        end
        if length(Start_ITD)>1
            nelerror('You can only specify 1 Starting ITD at a time for this template');
            err = 1;
        end
    end
end
function [Lsamps,Rsamps,dur_real,IACt] = GetOSCORNoiseBand(params)

global ChanConfigInfo %This global variable is set by the user at startup, and determines which channel is considered "contra-lateral".... taking into account the recording side, and the connections between channels and ears (i.e., this should be robust to someone mistakenly putting the right speaker in the left ear).
if isempty(ChanConfigInfo)
    ChanConfigInfo.ContraChan = 1;
end


%Gather all the required parameters

RandSeed = params.Inloop.Noise_Seed;
ITD = params.Inloop.ITD;
fLow = params.Inloop.Low_Cut_Freq*1000;
fHigh = params.Inloop.High_Cut_Freq*1000;
dur = params.Gating.Duration;
gateDur = params.Gating.Rise_fall_time;
fm = params.Inloop.OSCORfm_Start:params.Inloop.OSCORfm_Step:params.Inloop.OSCORfm_End;
OSCORtype = params.Inloop.IAC_Modulation_Type;
MSeq_N = params.Inloop.MSeq_N;
MSeq_StepDuration = params.Inloop.MSeq_StepDuration;
%-----------------------------------
Fsample = 48828.125;%Please don't edit this.
%-----------------------------------


%Pass the parameters to the moving noise function
[A, AB,dur_real,IACt] = MakeOSCORNoiseStim(OSCORtype,fLow,fHigh,dur,Fsample,RandSeed,ITD,fm,MSeq_N,MSeq_StepDuration,ChanConfigInfo.ContraChan);


if strcmp(params.Inloop.Interleave_Channels,'Yes')
    for i =1:length(A)
        tempA{(i-1)+1} = A{i};
        tempA{(i-1)+2} = AB{i};
        tempAB{(i-1)+1} = AB{i};
        tempAB{(i-1)+2} = A{i};
    end
    A = tempA;
    AB = tempAB;
    clear tempA tempAB;
end

%Do the gating
if ~strcmp(OSCORtype,'M-Sequence')
    Nsamples = length(A{1});
    gatesamps = cos(linspace(pi/2,pi,ceil(gateDur/1000*Fsample))).^2;
    changate = [gatesamps ones(1,Nsamples-2*ceil(gateDur/1000*Fsample)) fliplr(gatesamps)];
    for i = 1:length(A)
        Lsamps{i} = A{i}.*changate';
        Rsamps{i} = AB{i}.*changate';
    end
else
    for i = 1:length(A)
        Lsamps{i} = A{i};
        Rsamps{i} = AB{i};
    end
end
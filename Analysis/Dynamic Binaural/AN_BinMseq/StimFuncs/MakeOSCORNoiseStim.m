function [Aout,AB,dur_real,IACt] = MakeOSCORNoiseStim(OSCORtype,fLow,fHigh,dur,Fsample,RandSeed,ITD,fm,N,Seq_Step_Duration,ContraChan)

% fLow: lowest frequency component (Hz)
% fHigh: Highest frequency component (Hz)
% dur: duration of the sound (ms)
% Fsample: sampling frequency (Hz)
% RandSeed: random seed for noise generator
% ITD: ITD (ms)
% gateDur: duration of the cos2 gate to use (ms)
% Rho: initally zero..... then we do some mixing
% nChan: number of channels (2)

%% MakeOSCORNoiseStim
%First make the noise bands with the starting ITD applied
if strcmp(OSCORtype,'Sinusoid') || strcmp(OSCORtype,'Noise')
    [A,B] = MakeGaussNoiseBand(fLow,fHigh,dur, Fsample, 2, 0, ITD, 0, RandSeed,ContraChan);
end

%Now apply the oscillating inter-aural correlation
switch OSCORtype
    case 'Sinusoid'
        dt = 1/Fsample;
        t = 0:dt:(dur/1000)-dt;
        for i = 1:length(fm)
            eA = sin(2*pi*fm(i).*t)';
            eB = cos(2*pi*fm(i).*t)';
            
            Amod = A.*eA;
            Bmod = B.*eB;
            
            AB{i} = Amod+Bmod;
            Aout{i} = A;
        end
        dur_real = dur;
        
        IACt = eA;
    case 'Noise'
        %Now apply the oscillating inter-aural correlation
        [nbn] = MakeGaussNoiseBand(1000/dur,fm,dur, Fsample, 1,0, 0, 0, RandSeed+1,1);
        nbn = nbn/max(abs(nbn));
        nbn(:,2) = ((1-(nbn(:,1).^2))).^0.5;%See Grantham and Wightman (1979) JASA 65, 1509-1517
        Amod = A.*nbn(:,1);
        Bmod = B.*nbn(:,2);
        
        AB{1} = Amod+Bmod;
        Aout{1} = A;
        
        dur_real = dur;
        
        IACt = nbn(:,1);
    case 'M-Sequence'
        %Generate an MLS 2^N-1 long
        mseq = mls_mark(N,0,RandSeed);
        %The number of samples for each ITD token
        SegSamps = round(Seq_Step_Duration*1e-3*Fsample);
        
        ChunkSamps = SegSamps*length(mseq);
        
        %The number of mseq reps (Chunks) to be presented
        nChunks = max(2,round((Fsample*dur*1e-3)/ChunkSamps));
        
        %Make sure we end up with an even number of samples (makes all the Freq.
        %domain stuff easier)
        if rem(nChunks,2)
            nChunks = nChunks+1;
        end
        
        %The number of ITD tokens to be presented
        nSteps = nChunks*length(mseq);
        
        mseq = repmat(mseq,1,nChunks);
        
        %work out the real duration of the signal
        dt = 1/Fsample;
        
        dur_real = nSteps*SegSamps*dt*1000;
        
        [w1(:,1),w1(:,2)] = MakeGaussNoiseBand(0,Fsample/2, dur_real,Fsample, 2, -1, ITD, 0, RandSeed, ContraChan);
        [w2(:,1),w2(:,2)] = MakeGaussNoiseBand(0,Fsample/2, dur_real,Fsample, 2, 1, ITD, 0, RandSeed+1, ContraChan);
        
        
        w = w1;
        
        IACt = -ones(length(w),1);
        
        for i = 1:length(mseq)
            if mseq(i)==1
                w((i-1)*SegSamps+1: (i-1)*SegSamps+SegSamps,:) = w2((i-1)*SegSamps+1: (i-1)*SegSamps+SegSamps,:);
                IACt((i-1)*SegSamps+1: (i-1)*SegSamps+SegSamps) = 1;
            end
        end
        
        
        %apply an AM to obscure switches the the fine structure
        
        t = 0:dt:(dur_real/1000)-dt;
        fm = 1/(SegSamps*dt);
        
        env = 0.5*(1+sin(2*pi*fm.*t-(pi/2)));
        
        w(:,1) = w(:,1).*env';
        w(:,2) = w(:,2).*env';
        
        Aout{1} = w(:,1);
        AB{1} = w(:,2);
        
        clear w w1 w2 env t;
end

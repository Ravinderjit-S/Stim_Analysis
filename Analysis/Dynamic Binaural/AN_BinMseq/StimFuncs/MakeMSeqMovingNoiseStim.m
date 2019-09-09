function [w,ITDt,dur_real] = MakeMSeqMovingNoiseStim(dur,Fsample,RandSeed,mseq_ITDstart,mseq_ITDend,Rho,Seq_Step_Duration,N,ContraChan)


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

[w1(:,1),w1(:,2)] = MakeGaussNoiseBand(0,Fsample/2, dur_real,Fsample, 2, Rho, mseq_ITDstart, 0, RandSeed, ContraChan);
[w2(:,1),w2(:,2)] = MakeGaussNoiseBand(0,Fsample/2, dur_real,Fsample, 2, Rho, mseq_ITDend, 0, RandSeed+1, ContraChan);


w = w1;

ITDt = ones(length(w),1)*mseq_ITDstart;

for i = 1:length(mseq)
    if mseq(i)==1
        w((i-1)*SegSamps+1: (i-1)*SegSamps+SegSamps,:) = w2((i-1)*SegSamps+1: (i-1)*SegSamps+SegSamps,:);
        ITDt((i-1)*SegSamps+1: (i-1)*SegSamps+SegSamps) = mseq_ITDend;
    end
end


%apply an AM to obscure switches the the fine structure

t = 0:dt:(dur_real/1000)-dt;
fm = 1/(SegSamps*dt);

env = 0.5*(1+sin(2*pi*fm.*t-(pi/2)));

w(:,1) = w(:,1).*env';
w(:,2) = w(:,2).*env';


clear w1 w2 env t;
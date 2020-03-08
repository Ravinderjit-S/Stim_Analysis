function [stim,mseqAM] = AM_mseq(m,upperF,fs)
%m = bits of mseq
%upperF = highest frequency to characterize with mseq
%fs = sampling rate


Point_dur = 1/upperF;
mseq = mls(m,0);
mseq = (mseq+1)/2; %make mseq bounce b/t 0 & 1

Point_dur = 1/upperF; %duration of a single point in the mseq
mseqAM = [];
for i =1:length(mseq)
    mseqAM = [mseqAM ones(1,round(Point_dur*fs))*mseq(i)];
end
w = dpss(round(Point_dur*fs),1,1);
w = w-w(1); w = w/max(w); %dpss ramp so energy in modulation doesn't spread out far beyond upperF

mseqAM = conv(mseqAM,w,'same');
mseqAM = mseqAM / max(mseqAM);

nn = randn(1,length(mseqAM));
stim = nn.*mseqAM;

end




function [stim, mseqFM] = FM_mseq(f1,f2,m,upperF,fs)
%f1,f2 = frequencies to bounce between
%m = bits of mseq
%upperF = highest frequency to characterize with mseq
%fs = sampling rate

Point_dur = 1/upperF;
mseq = mls(m,0);

mseqFM = [];
for i = 1:length(mseq)
    mseqFM = [mseqFM ones(1,round(Point_dur*fs))*mseq(i)];
end

RealUpperF = fs/round(Point_dur*fs);

t = 0:1/fs:length(mseqFM)/fs -1/fs;
x1 = sin(2*pi*f1.*t);
x1(mseqFM ==-1) = 0;
x2 = sin(2*pi*f2.*t);
x2(mseqFM==1) = 0;
AM = 0.5* (sin(2*pi.*t*RealUpperF-pi/2)+1); %AM to mask phase discontinuities

stim = AM .*(x1 + x2);

end
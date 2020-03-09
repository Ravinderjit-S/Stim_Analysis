function [stim, mseqIAC] = IAC_mseq(m, upperF, fs)

Point_dur = 1/upperF;

mseq = mls(m,0);

mseqIAC = [];
for i =1:length(mseq)
    mseqIAC = [mseqIAC ones(1,round(Point_dur*fs))*mseq(i)];
end
RealUpperF = fs/round(Point_dur*fs);

stim = randn(1,length(mseqIAC));
stim(2,:) = stim(1,:) .* mseqIAC;

end
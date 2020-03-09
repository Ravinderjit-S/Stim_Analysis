clear

fs = 48828;

upperF = 1000;
Point_dur = 1/upperF;

mseq = mls(11,0);

mseqIAC = [];
for i =1:length(mseq)
    mseqIAC = [mseqIAC ones(1,round(Point_dur*fs))*mseq(i)];
end
RealUpperF = fs/round(Point_dur*fs);

nn = randn(1,length(mseqIAC))

nn(2,:) = nn(1,:) .* mseqIAC;

t = 0:1/fs:length(mseqIAC)/fs - 1/fs;
figure,plot(t,nn'), hold on
plot(t,mseqIAC,'k')

soundsc(nn,fs)



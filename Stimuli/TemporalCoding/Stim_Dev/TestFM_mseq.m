clear

fs = 48828;
f1 = 4000;
f2 = 1.05*f1;
upperF = 1000;
Point_dur = 1/upperF;

mseq = mls(11,0);

mseqFM = [];
for i =1:length(mseq)
    mseqFM = [mseqFM ones(1,round(Point_dur*fs))*mseq(i)];
end

RealUpperF = fs/round(Point_dur*fs);

t = 0:1/fs:length(mseqFM)/fs -1/fs;

x1 = sin(2*pi*f1.*t);
x1(mseqFM==-1) = 0;
x2 = sin(2*pi*f2.*t);
x2(mseqFM==1) = 0;
AM = 0.5 * (sin(2*pi.*t*RealUpperF-pi/2)+1);
stim = AM.*(x1+x2);


figure,plot(t,stim,'b',t,mseqFM,'r',t,AM,'k')
ylim([-1.1 1.1])

soundsc(stim,fs)







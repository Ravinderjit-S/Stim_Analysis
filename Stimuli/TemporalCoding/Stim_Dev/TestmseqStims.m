clear
load('mseqEEG.mat')

range_carrier = [4000 6000];

stimAM  = AM_mseq(mseqEEG,Point_len);
stimFM  = FM_mseq(range_carrier,mseqEEG,Point_len,fs);
stimIAC = IAC_mseq(mseqEEG);


t = 0:1/fs:length(mseqEEG)/fs -1/fs;

figure,plot(t,stimAM',t,mseqEEG,'k'),title('AM')
figure,plot(t,stimFM',t,mseqEEG,'k'),title('FM')
figure,plot(t,stimIAC',t,mseqEEG,'k'),title('IAC')


soundsc(stimAM,fs)
pause(t(end) + 0.5)

soundsc(stimFM,fs)
pause(t(end) + 0.5)

soundsc(stimIAC,fs)


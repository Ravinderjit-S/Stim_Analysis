clear

load('mseqEEG_80.mat')
%load('/media/ravinderjit/Storage2/EEGdata/mseqEEG_80.mat')

fs= 48828;

resample_fs = 4096;
m2 = resample(mseqEEG,4096,fs);
m2(m2<0) = -1;
m2(m2>0) = 1;

t = 0:1/fs:length(mseqEEG)/fs -1/fs;
t2 = 0:1/4096:length(m2)/4096-1/4096;
figure,plot(t,mseqEEG,'b',t2,m2,'r')

mseqEEG_4096 = m2;
save('mseqEEG_80_4096.mat','mseqEEG_4096','Point_len')



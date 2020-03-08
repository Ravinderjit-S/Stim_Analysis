clear
path = '../../CommonExperiment';
p = genpath(path);
addpath(p);

path = '../';
p = genpath(path);
addpath(p);

fs = 48828;
tic()
[stim,mseqAM] = AM_mseq(11,1000,fs);
toc()
t = 0:1/fs:length(stim)/fs-1/fs;

figure,plot(t,stim,t,mseqAM,'r')
soundsc(stim,fs)





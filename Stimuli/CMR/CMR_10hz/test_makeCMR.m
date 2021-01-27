% test make CMRstim
clear
path = '../../CommonExperiment';
addpath(genpath(path));

fc = 4000;
fs = 48828.125;
fm = 10;
ofmbw = 1;
ofSNR = 12;
flankdist = 2;
flankbw = 1;
condition = 2;
dur = 4;
ramp = 0.01;
target_modf = 223;

tic()
x = makeCMRstim_mod(fc, fs, fm, ofmbw, ofSNR, flankdist, flankbw,...
    condition, dur, ramp,target_modf);
toc()

t = 0:1/fs:length(x)/fs-1/fs;

figure,plot(t,x)

figure,
spectrogram(x, round(0.05*fs),round(0.9*0.05*fs),2000:6000,round(fs),'yaxis')

figure,
pmtm(x,2.5,1:6000,fs)


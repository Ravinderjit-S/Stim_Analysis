clear
path = '../CommonExperiment';
addpath(genpath(path));

fs = 48828;
tlen = 2;
t = 0:1/fs:tlen-1/fs;

ERB_halfwidth = 0.5;
ERBspacing = 1.5;
target_f = 4000;
noise_bands = CMRbands(target_f, ERB_halfwidth, ERBspacing);

SNRdb = 0;
mod_band = [2 10];
target_modf = 40;

coh = 1;
bp_mod_fo = 1/2 * 5 *fs; %filter order for slowest modulation instance ... keep same sharpness for all modulation filters so setting here

tic()
[Sig] = CMR_randMod(noise_bands,target_f,SNRdb,mod_band,target_modf,fs,tlen,coh,bp_mod_fo);
toc()

soundsc(Sig,fs)

figure,pmtm(Sig,2.5,[],fs)
figure,spectrogram(Sig,round(0.02*fs),round(0.02*fs*.8),2000:1:7000,fs,'yaxis')



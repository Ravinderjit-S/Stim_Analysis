clear
path = '../CommonExperiment';
addpath(genpath(path));

fs = 48828;
tlen = 1;
t = 0:1/fs:tlen-1/fs;

ERB_halfwidth = 0.5;
ERBspacing = 1.5;
target_f = 4000;
noise_bands = CMRbands(target_f, ERB_halfwidth, ERBspacing);

SNRdb = 0;
mod_band = [4 24];
target_modf = 0;

coh = 1;
% 
% tic()
% [Sig] = CMR_randMod(noise_bands,target_f,SNRdb,mod_band,target_modf,fs,tlen,coh);
% toc()

tic()
[Sig] = CMR_randMod_clicky(noise_bands,target_f,SNRdb,mod_band,target_modf,fs,tlen,coh);
toc()

soundsc(Sig,fs)

figure,pmtm(Sig,2.5,[],fs)
figure,spectrogram(Sig,round(0.02*fs),round(0.02*fs*.8),2000:1:7000,fs,'yaxis')

%% 3AFC version
tic()
[Sig,answer] = CMR_randMod_clicky_3AFC(noise_bands,target_f,SNRdb,mod_band,target_modf,fs,tlen,1,.050,.3);
toc()

soundsc(Sig,fs)

figure,
plot(Sig')

figure,spectrogram(Sig(1,:),round(0.02*fs),round(0.02*fs*.8),2000:1:7000,fs,'yaxis')

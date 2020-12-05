clear
path = '../../CommonExperiment';
addpath(genpath(path));

fs = 48828;
tlen = 1;
t = 0:1/fs:tlen-1/fs;

ERB_halfwidth = 0.5;
ERBspacing = 1.5;
target_f = 4000;
noise_bands = CMRbands(target_f, ERB_halfwidth, ERBspacing);
SNRdb = -6;
mod_band = [2 10];
target_modf = 0;
coh = 1;
bp_mod_fo = 1/2 * 5 *fs; %filter order for slowest modulation instance ... keep same sharpness for all modulation filters so setting here

tic()
[Sig] = CMR_randMod_3AFC(noise_bands,target_f,SNRdb,mod_band,target_modf,fs,tlen,coh,bp_mod_fo);
toc()

silence = zeros(1,round(0.5*fs));
order = randperm(3);
for i =1:3
    x = Sig{order(i)};
    energy = rms(x);
    y(i,:) = rampsound(x,fs,.050)/energy;
    y(i,:) = scaleSound(y(i,:)); 
end
all_stim = [y(1,:) silence y(2,:) silence y(3,:)];
figure,plot(all_stim)
soundsc(all_stim,fs);

figure,pmtm(Sig{3},2.5,[],fs)
figure,spectrogram(Sig{3},round(0.02*fs),round(0.02*fs*.8),0:1:6000,fs,'yaxis')




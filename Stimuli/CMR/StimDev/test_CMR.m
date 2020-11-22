clear
path = '../CommonExperiment';
addpath(genpath(path));

fs = 44100;
tlen = 1;
t = 0:1/fs:tlen-1/fs;
upper_f = 4000;
target_f = 2000;
lower_f = 1000;
noise_half_bw = 200; %half the bandwidth of noise. noise is centered on lower and upper f
center_freqs = [lower_f target_f upper_f];
SNRdb = 0;
n_mod_cuts = [118 125];
target_modf = 4;
phase = 180;
coh = 0;
bp_mod_fo = 1/2 * 5 *fs; %filter order for slowest modulation instance ... keep same sharpness for all modulation filters so setting here

tic()
[Sig] = CMR_randMod(center_freqs,noise_half_bw,SNRdb,n_mod_cuts,target_modf,fs,tlen,coh,bp_mod_fo);
toc()

order = randperm(3);
for i =1:3
    x = Sig{order(i)};
    x = rampsound(x,fs,.050);
    x = scaleSound(x);
    soundsc(x,fs);
    pause(tlen+0.5*tlen);
end

figure,pmtm(Sig{3},2.5,[],fs)
figure,spectrogram(Sig{3},round(0.02*fs),round(0.02*fs*.8),0:1:6000,fs,'yaxis')




clear
fs = 48828;
center_f = 4000;
ERB_halfwidth = 0.5;
ERBspacing = 1.5;
noise_bands = CMRbands(center_f, ERB_halfwidth, ERBspacing);
mod_band = [101,109];
mod_bpfo = 1/2*5*fs;
t_coh = 1;
t_incoh = 1;

tic()
[Sig,mods] = Noise_coherenceABAB(noise_bands,mod_band,mod_bpfo,t_coh,t_incoh,fs);
toc()

soundsc(Sig,fs)

t = 0:1/fs:length(Sig)/fs-1/fs;
figure,plot(t,Sig)
figure,spectrogram(Sig,round(.01*fs),round(.01*fs*.9),1:7000,fs,'yaxis')

figure,plot(t,mods')



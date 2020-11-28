clear
fs = 48828;
center_f = 4000;
ERB_halfwidth = 0.5;
ERBspacing = 1.5;
noise_bands = CMRbands(center_f, ERB_halfwidth, ERBspacing);
mod_band = [2,10];
mod_bpfo = 1/2 * 5 *fs;
t_len = 1;

tic()
[Sig] = Noise_coherence_3AFC(noise_bands,mod_band,mod_bpfo,t_len,fs);
toc()


figure, 
spectrogram(Sig{1},round(0.01*fs),round(0.01*fs*0.90),2000:7000,fs,'yaxis')
figure,
spectrogram(Sig{2},round(0.01*fs),round(0.01*fs*0.90),2000:7000,fs,'yaxis')
figure,
spectrogram(Sig{3},round(0.01*fs),round(0.01*fs*0.90),2000:7000,fs,'yaxis')

t = 0:1/fs:t_len-1/fs;

figure,
subplot(3,1,1),plot(t,Sig{1})
subplot(3,1,2),plot(t,Sig{2})
subplot(3,1,3),plot(t,Sig{3})
linkaxes



order = randperm(3);

for j = 1:3
    soundsc(Sig{order(j)},fs)
    pause(t_len*1.5)
end

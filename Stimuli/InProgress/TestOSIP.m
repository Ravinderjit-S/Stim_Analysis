clear

dur = 1;
fs = 48828;
f = 1000;
fm = 24;
noise = 0;

stim = OSIP(dur,fs,f,fm,noise);

soundsc(stim,fs)
t = 0:1/fs:1-1/fs;
figure,plot(t,stim')
ylim([-1.1 1.1])
xlim([0 2/fm])

figure,
spectrogram(stim(2,:), round(0.03*fs),round(0.02*fs*0.9),[],fs,'yaxis'), ylim([0,5])



soundsc(stim,fs)

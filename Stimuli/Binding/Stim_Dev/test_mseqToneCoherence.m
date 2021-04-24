clear

load('mseqAM_150_bits10.mat');
fs = 48828;

[Tones_f, ERBspace] = Get_Tones(8,2,1500,8000);

tone_freq = Tones_f(1:8);

stim = mseq_tone_coherence(tone_freq,fs,mseqAM);
t = 0:1/fs:length(stim)/fs-1/fs;

figure,plot(t,stim);
soundsc(stim,fs);




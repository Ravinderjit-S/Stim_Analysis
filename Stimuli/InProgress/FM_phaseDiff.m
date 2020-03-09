
fs = 48828;
t = 0:1/fs:1-1/fs;

f1 = 1000;
f2 = 4000;


dF1 = f1*.1;
dF2 = f2*.1;
fm = 4;
phase_deg = 90;
phase_diff = (phase_deg / 360) * 2*pi;

phase1 = dF1/fm * sin(2*pi*fm*t);
phase2 = dF2/fm * sin(2*pi*fm*t);

phase3 = dF2/fm * sin(2*pi*fm*t + phase_diff);

x1 = sin(2*pi*f1*t + phase1);
x2 = sin(2*pi*f2*t + phase2);
x3 = sin(2*pi*f2*t + phase3);

sig  = x1 + x2;
sig2 = x1 + x3; 

soundsc(sig,fs)
pause(1.3)
soundsc(sig2,fs)
pause(1.3)
soundsc(sig,fs)


% figure,plot(t,sig)
figure,spectrogram(sig, round(0.03*fs), round(0.02*fs*0.9),[],fs,'yaxis'), ylim([0,16])

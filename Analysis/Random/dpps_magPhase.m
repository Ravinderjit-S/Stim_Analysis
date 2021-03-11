clear

dur = 1;
fs = 44100;
TBW = 3;

dpss_seq = dpss(dur*fs,TBW);


dpss_phase = unwrap(angle(fft(dpss_seq)));
f = 0:1:fs-1;
figure,
plot(f,dpss_phase)

figure,
plot(dpss_seq)






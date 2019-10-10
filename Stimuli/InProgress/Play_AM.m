fs = 40000;
noise = randn(1,fs);
filt = designfilt('bandpassfir', 'StopbandFrequency1', 100, 'PassbandFrequency1', 200, 'PassbandFrequency2', 1500, 'StopbandFrequency2', 2000, 'StopbandAttenuation1', 60, 'PassbandRipple', 1, 'StopbandAttenuation2', 60, 'SampleRate', fs);
noise = filter(filt,noise);


fm = 8;
t = 0:1/fs:1-1/fs;
A = 0.5*(cos(2*pi*fm.*t)+1); 
tone = sin(2*pi*800*t);

sig = A.*noise;
sig2 = A.*tone;

soundsc(sig,fs)
soundsc(sig2,fs)



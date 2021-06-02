clear

[s1,fs1] = audioread('NCM016_05-01.wav');
[s2,fs2] = audioread('PNF139_05-03.wav');

fs_final = 48828;

s1_resamp = resample(s1,fs_final,fs1);
s2_resamp = resample(s2,fs_final,fs2);

soundsc(s1,fs1)
pause(4.0)
soundsc(s1_resamp,fs_final)
pause(4.0)
soundsc(s2,fs2)
pause(4.0)
soundsc(s2_resamp,fs_final)

t1 = 0:1/fs1:length(s1)/fs1 - 1/fs1;
t2 = 0:1/fs_final:length(s1_resamp)/fs_final-1/fs_final;
figure, hold on
plot(t1,s1)
plot(t2,s1_resamp)

sentences = {s1_resamp,s2_resamp};
fs = 48828;

save('Sentences_48828.mat','sentences','fs')


clear

f0 = 80;
dur = 0.4;
sequence = [f0, f0* (2^(1/6)), f0, f0*(2^(5/12)), f0 * (2^(1/3))];

fs = 44100;
[y,d] = lowpass(randn(1,fs*dur),5000,fs);

stim_os = [];


for s = 1:length(sequence)
    stim_s = OSCOR(dur,fs,sequence(s),d,1);
    stim_os = [stim_os stim_s];
end

stim_am = [];
t =0:1/fs:dur-1/fs;
for s =1:length(sequence)
    stim_s = (0.5*sin(2*pi*sequence(s) .* t)+0.5) .* y;
    stim_am = [stim_am, stim_s];
end

stim_os = [stim_os,stim_os,stim_os,stim_os];
stim_am = [stim_am,stim_am,stim_am,stim_am];

soundsc(stim_os,fs)
%pause(dur*length(sequence)+dur)
soundsc(stim_am,fs)

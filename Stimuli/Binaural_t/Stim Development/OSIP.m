function [stim] = OSIP(dur,fs,f,fm,noise)
%dur = duration in sec
%fs = sampling rate
%fm = frequency of OSIP
%noise = if 1, use white noise

t = 0:1/fs:dur-1/fs;

stimL = sin(2*pi*f.*t);
stimR = sin(2*pi*f.*t + pi/2*sin(2*pi*fm.*t)+pi/2);
stim = [stimL;stimR];

%% make noise with bunch of tones ... kinda wonky
if noise == 1
    phases = 2*pi*rand(1,floor(fs/2*dur));
    freqs = 0:1/dur:floor(fs/2);
    stimL = zeros(1,dur*fs);
    stimR = zeros(1,dur*fs);
    for n = 1:length(phases) 
        stimL = stimL + sin(2*pi*freqs(n).*t + phases(n));
        stimR = stimR + sin(2*pi*freqs(n).*t + phases(n) + 2*pi*sin(2*pi*fm.*t));
    end
    stim = [stimL/length(phases);stimR/length(phases)];
end



    





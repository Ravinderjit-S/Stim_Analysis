function [stim] = mseq_tone_coherence(tone_freq,fs,mseq)


t = 0:1/fs:length(mseq)/fs-1/fs;
stim = zeros(1,length(t));
for tn = 1:length(tone_freq)
    tone = sin(2*pi*tone_freq(tn).*t);
    stim = stim + tone.*mseq;
end


    






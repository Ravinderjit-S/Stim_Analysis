% generate AM mseq for experiment

bits = 11;
upperF = 1000;
[stim,mseqAM] = AM_mseq(bits,upperF,fs);
stim = [stim;stim];

save('AMmseqStim.mat','stim','mseqAM')




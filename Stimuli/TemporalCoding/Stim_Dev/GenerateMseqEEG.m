% generate stim for temporal coding experiments
clear
bits = 11;
upperF = 1000;
fs = 48828;

[mseqEEG,Point_len] = EEGmseq(bits,upperF,fs);

%save('mseqEEG.mat','mseqEEG','bits','fs','Point_len')


upperF = 500;
[mseqEEG_500,Point_len] = EEGmseq(bits,upperF,fs);
%save('mseqEEG_500.mat','mseqEEG_500','bits','fs','Point_len')





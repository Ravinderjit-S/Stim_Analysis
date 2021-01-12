% generate stim for temporal coding experiments
clear
bits = 10;
upperF = 80;
fs = 48828;

[mseqEEG,Point_len] = EEGmseq(bits,upperF,fs);

save(['mseqEEG_' num2str(upperF) '.mat'],'mseqEEG','bits','fs','Point_len')


% upperF = 500;
% [mseqEEG_500,Point_len] = EEGmseq(bits,upperF,fs);
%save('mseqEEG_500.mat','mseqEEG_500','bits','fs','Point_len')





% generate stim for temporal coding experiments
clear
bits = 11;
upperF = 1000;
fs = 48828;

[stimAM, mseqAM] = AM_mseq(bits,upperF,fs);
stimAM = [stimAM;stimAM];
save('AMmseqStim.mat','stimAM','mseqAM')

f1 = 4000;
f2 = 1.05*f1;
[stimFM, mseqFM] = FM_mseq(f1,f2,bits,upperF,fs);
stimFM = [stimFM;stimFM];
save('FMmseqStim.mat','stimFM','mseqFM')

[stimIAC, mseqIAC] = IAC_mseq(bits,upperF,fs);
save('IACmseqStim.mat','stimIAC','mseqIAC')




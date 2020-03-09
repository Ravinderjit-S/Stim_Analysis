function [mseqEEG,Point_len] = EEGmseq(m,upperF,fs)

Point_len = floor(fs/upperF);
mseq = mls(m,0);

mseqEEG = [];
for i =1:length(mseq)
    mseqEEG = [mseqEEG ones(1,Point_len)*mseq(i)];
end

end
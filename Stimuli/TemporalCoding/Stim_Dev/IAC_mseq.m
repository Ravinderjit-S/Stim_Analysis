function [stim] = IAC_mseq(mseqIAC)

stim = randn(1,length(mseqIAC));
stim(2,:) = stim(1,:) .* mseqIAC;

end
load('mseqEEG_150_bits10.mat')

fig_path = '/media/ravinderjit/Data_Drive/Data/Figures/TemporalCoding/';

stim = AM_mseq(mseqEEG, Point_len);
t = 0:1/fs:length(stim)/fs - 1/fs;

stim = stim(1,:);
stim = stim / max(abs(stim));

mseqAM = mseqEEG;
mseqAM = (mseqAM+1)/2; %make mseq bounce b/t 0 & 1
w = dpss(Point_len,1,1);
w = w-w(1); w = w/max(w); %dpss ramp so energy in modulation doesn't spread out far beyond upperF

mseqAM = conv(mseqAM,w,'same');
mseqAM = mseqAM / max(mseqAM);


figure, hold on
plot(t,stim)
plot(t,mseqAM,'k')
xlim([0,0.5])
ylim([-1,1.1])

print([fig_path 'AMmseq.svg'], '-dsvg')





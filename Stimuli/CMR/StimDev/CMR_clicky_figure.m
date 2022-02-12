clear
path = '../CommonExperiment';
addpath(genpath(path));

fs = 48828;
tlen = 1;
t = 0:1/fs:tlen-1/fs;

ERB_halfwidth = 0.5;
ERBspacing = 1.5;
target_f = 4000;
noise_bands = CMRbands(target_f, ERB_halfwidth, ERBspacing);

SNRdb = 0;
mod_band = [4 24];
target_modf = 0;

coh = 1;

tic()
[Sig] = CMR_randMod_clicky(noise_bands,target_f,SNRdb,mod_band,target_modf,fs,tlen,coh);
toc()

coh = 0;

tic()
[Sig_0] = CMR_randMod_clicky(noise_bands,target_f,SNRdb,mod_band,target_modf,fs,tlen,coh);
toc()

full_sig = [Sig_0, zeros(1, round(fs*0.2)), Sig];

fig_path = '/home/ravinderjit/Dropbox/Figures/MTBproj/';

%figure,pmtm(Sig,2.5,[],fs)
figure,spectrogram(full_sig,round(0.02*fs),round(0.02*fs*.8),2000:1:7000,fs,'yaxis')
yticks([3,4,5])
ylim([2.5, 6])
xticks([0.01, 1, 1.2, 2.19])
xticklabels({'0','1','0','1'})
set(gca,'fontsize',15)
fig = gcf;
fig.PaperUnits = 'inches';
fig.PaperPosition = [0 0 14 6];
print([fig_path 'CMR_clicky'],'-dsvg')


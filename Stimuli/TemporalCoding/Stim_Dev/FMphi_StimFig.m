clear

fs = 44100;
f1 = 500;
f2 = 4*f1;
stim_dur = 1;
fm = 8;
phi = 180;
diotic = 0;
ref = 0;

fig_path = '/media/ravinderjit/Data_Drive/Data/Figures/TemporalCoding/FMphi/';

stim = FM_phi(f1,f2,fs,stim_dur,fm,phi,diotic,ref);
stim = stim{3};
t = 0:1/fs:1-1/fs;
figure,
plot(t,stim(1,:))

fig = figure;
spectrogram(stim(1,:), round(.050*fs), round(0.9 * 0.050*fs),100:2500,fs,'yaxis')
colorbar('off')
xticks([100, 500, 900])
yticks([0.5, 1, 1.5, 2])
set(gca, 'fontsize',12)
print(fig, [fig_path 'FMphiStim.svg'],'-dsvg')
print(fig, [fig_path 'FMphiStim.eps'],'-depsc')



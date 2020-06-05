% Making Figure to Show Stimulus

fs = 48828;
Corr_inds = [6:12];
f_start = 100;
f_end = 8000;
Tones_num = 16;
ERB_spacing = []; %if specified, takes precedence over tones_num
FigPath = '/media/ravinderjit/Data_Drive/Data/Figures/BindingPilot/';

[stimABAB, envs, ERBspace, Tones_f] = Stim_Bind_ABAB(Corr_inds, fs, f_start, f_end, Tones_num, ERB_spacing);


figure()
%subplot(2,1,1)
t = 0:1/fs:length(stimABAB)/fs-1/fs;
stimABAB = stimABAB / max(abs(stimABAB));
plot(t,stimABAB),hold on
xlim([0,4])
set(gca,'Xtick',[1,2,3,4])
set(gca,'Xticklabel',[])
% ylabel('Amplitude')
set(gca,'fontsize',15)
set(gca,'Ytick', [-1, 0, 1])
set(gca,'Yticklabel',[])
fig = gcf;
fig.PaperUnits = 'inches';
fig.PaperPosition = [0 0 4 1.25];
print([FigPath 'stimABABtime'],'-dpng','-r0')

% subplot(2,1,2)
figure()
spectrogram(stimABAB,fs/50,[],[],fs,'yaxis'), ylim([0.08 8.4])
colorbar('off')
set(gca,'fontname','Arial')
set(gca,'fontsize',11)
set(gca,'Ytick', [2, 4, 6, 8])
set(gca,'Xtick',[1,2,3,4])
xlim([0,4])
fig = gcf;
fig.PaperUnits = 'inches';
fig.PaperPosition = [0 0 4 2];

print([FigPath 'stimABAB'],'-dpng','-r0')

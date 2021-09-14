fs = 48828;
Corr_inds = [15:20];
f_start = 200;
f_end = 8000; 
Tones_num = 20;
ERB_spacing = []; %if specified, takes precedence over Tones_num


tic()
[stimABAB, envs, ERBspace, Tones_f] = Stim_Bind_ABABA(Corr_inds, fs, f_start, f_end, Tones_num, ERB_spacing);
toc()

Corr_inds = [1,4,8,12,16,20];

tic()
[stimABAB_2, envs, ERBspace, Tones_f] = Stim_Bind_ABABA(Corr_inds, fs, f_start, f_end, Tones_num, ERB_spacing);
toc()

t = 0:1/fs:length(stimABAB)/fs - 1/fs;

figure
subplot(2,2,1),
plot(t,stimABAB)
%plot(t,abs(hilbert(stimABAB)))
xlim([0,4])
ylim([-6,6])
xticks([0,1,2,3,4])
yticks([-6,-3,0,3,6])

subplot(2,2,3)
[s,f,t_s] = spectrogram(stimABAB,round(fs/50),round(0.9 * fs/50),[],fs);
imagesc(t_s,fliplr(f)/1e3,pow2db(abs(s).^2),[-25,40])
set(gca,'YDir','normal')
ylim([0 9])
xlim([0,4])
xticks([0,1,2,3,4])
yticks([0,2,4,6,8])

subplot(2,2,2)
plot(t,stimABAB_2)
%plot(t,abs(hilbert(stimABAB_2)))
xlim([0,4])
ylim([-6,6])
xticks([0,1,2,3,4])

subplot(2,2,4)
[s,f,t_s] = spectrogram(stimABAB_2,round(fs/50),round(0.9 * fs/50),[],fs);
imagesc(t_s,fliplr(f)/1e3,pow2db(abs(s).^2),[-25,40])
set(gca,'YDir','normal')
ylim([0 9])
xlim([0,4])
xticks([0,1,2,3,4])
yticks([-6,-3,0,3,6])
yticks([0,2,4,6,8])



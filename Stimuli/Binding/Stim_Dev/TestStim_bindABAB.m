fs = 48828;
Corr_inds = [9:20];
f_start = 200;
f_end = 8000; 
Tones_num = 20;
ERB_spacing = []; %if specified, takes precedence over Tones_num

tic()
[stimABAB, envs, ERBspace, Tones_f] = Stim_Bind_ABABA(Corr_inds, fs, f_start, f_end, Tones_num, ERB_spacing);
toc()
tic()
Corr_inds = [];
[stimAAAA, envs, ERBspace, Tones_f] = Stim_Bind_ABABA(Corr_inds, fs, f_start, f_end, Tones_num, ERB_spacing);
toc()

t = 0:1/fs:length(stimABAB)/fs - 1/fs;

figure,
plot(t,stimABAB)

figure,
plot(t,envs)

soundsc(stimAAAA,fs)
pause(5.4)
soundsc(stimABAB,fs)


figure()
spectrogram(stimABAB,fs/50,[],[],fs,'yaxis'), ylim([0 10])

%save('Binding20tonesExample_12.mat', 'Corr_inds', 'envs', 'fs', 'stimABAB')


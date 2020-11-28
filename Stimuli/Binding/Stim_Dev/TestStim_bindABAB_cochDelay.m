fs = 48828;
%Corr_inds = [12:16];
Corr_inds = 1:2:16;
f_start = 100;
f_end = 8000; 
Tones_num = 16;
ERB_spacing = []; %if specified, takes precedence over Tones_num

tic()
[stimABAB, envs, ERBspace, Tones_f] = Stim_Bind_ABAB_cochDelay(Corr_inds, fs, f_start, f_end, Tones_num, ERB_spacing);
toc()
 
[stimABAB_noDel, ~, ERBspace, Tones_f] = Stim_Bind_ABAB(Corr_inds, fs, f_start, f_end, Tones_num, ERB_spacing);

Corr_inds = 9:16;
[stimAAAA, ~, ERBspace, Tones_f] = Stim_Bind_ABAB(Corr_inds, fs, f_start, f_end, Tones_num, ERB_spacing);




sound(stimAAAA,fs)
pause(4.4)
sound(stimABAB,fs)
pause(4.4)
sound(stimABAB_noDel,fs)


t = 0:1/fs:length(stimABAB)/fs-1/fs;




figure()
spectrogram(stimABAB,fs/50,[],[],fs,'yaxis'), ylim([0 10])

figure,
plot(t,envs)



fs = 48828;
%Corr_inds = [12:16];
Corr_inds = [1:12];
f_start = 100;
f_end = 8000; 
Tones_num = 16;
ERB_spacing = []; %if specified, takes precedence over Tones_num

tic()
[stimABAB, envs, ERBspace, Tones_f] = Stim_Bind_ABAB(Corr_inds, fs, f_start, f_end, Tones_num, ERB_spacing);
toc()
tic()
Corr_inds = [];
[stimAAAA, envs, ERBspace, Tones_f] = Stim_Bind_ABAB(Corr_inds, fs, f_start, f_end, Tones_num, ERB_spacing);
toc()


sound(stimAAAA,fs)
pause(4.4)
sound(stimABAB,fs)

t = 0:1/fs:2.8-1/fs;




figure()
spectrogram(stimABAB,fs/50,[],[],fs,'yaxis'), ylim([0 10])


fs = 48828;
%Corr_inds = [12:16];
Corr_inds1 = [5:16];
Corr_inds2 = [1:12];
f_start = 100;
f_end = 8000; 
Tones_num = 16;
ERB_spacing = []; %if specified, takes precedence over Tones_num

tic()
[stimABAB, envs, ERBspace, Tones_f] = Stim_Bind_AB1AB2(Corr_inds1, Corr_inds2, fs, f_start, f_end, Tones_num, ERB_spacing);
toc()


sound(stimABAB,fs)


figure()
spectrogram(stimABAB,fs/50,[],[],fs,'yaxis'), ylim([0 10])



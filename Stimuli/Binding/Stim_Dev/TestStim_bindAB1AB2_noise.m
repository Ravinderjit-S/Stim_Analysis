fs = 48828;
%Corr_inds = [12:16];
Corr_inds1 = [5:16];
Corr_inds2 = [1:12];
f_start = 100;
f_end = 8000; 
Tones_num = 16;
ERB_spacing = []; %if specified, takes precedence over Tones_num



[Tones_f, ERBspace] = Get_Tones(Tones_num, ERB_spacing, f_start, f_end); %returns tone frequencies 
tic()
for i = 1:length(Tones_f)
    f = Tones_f(i);
    noise_octaves = 0.1; %noise bandwidth will be this many octaves
    bandwidth = f*2^noise_octaves-f;
    BPfilt{i} = designfilt('bandpassfir', 'FilterOrder', fs/2,'CutoffFrequency1',f-bandwidth/2, ...
            'CutoffFrequency2',f+bandwidth/2,'SampleRate',fs);
end
toc()

tic()
[stimABAB, envs, ERBspace, Tones_f] = Stim_Bind_AB1AB2_noise(Corr_inds1, Corr_inds2, fs, f_start, f_end, Tones_num, ERB_spacing);
toc()


soundsc(stimABAB,fs)


figure()
spectrogram(stimABAB,fs/50,[],[],fs,'yaxis'), ylim([0 10])


%% Make Stims for Chinchilla Binding Experiment



%% Parameters

fs = 48828;
Corr_inds = {[1:12], [4:12]};
f_start = 300;
f_end = 18e3; 
ERB_spacing = 1.5; 
Tones_num = [];

Trials = 150;
conds = repmat([1,2],1,Trials);
conds = conds(randperm(length(conds)));


%% Gen Stims

A_stims = {};

for j = 1:length(conds)
    [stimABABA, envs, ERBspace, Tones_f] = Stim_Bind_ABABA_chin(Corr_inds{conds(j)}, fs, f_start, f_end, Tones_num, ERB_spacing);
    A_stims{j} = stimABABA;
end
   
%% Save Stuff
trigs = conds;
save('ChinBindingStims.mat','A_stims','trigs')
    
% soundsc(stimABABA,fs);
% 
% figure()
% spectrogram(stimABABA,fs/50,[],[],fs,'yaxis'), ylim([0 20])


%Generate wav files for bindng expereiment with AB1AB2 noise
clear
path = '../../CommonExperiment';
p = genpath(path);
addpath(p); %add path to commonly used functions
%% Stim & Experimental parameters
load('s.mat')
rng(s)

fs = 48828;
f_start = 100;
f_end = 8000; 
Tones_num = 16;
ERB_spacing = []; %if specified, takes precedence over Tones_num


Corr_inds{1,1} = 1:16; 
Corr_inds{1,2} = 1:16; 
Corr_inds{2,1} = 5:16;
Corr_inds{2,2} = 1:12;

nconds = length(Corr_inds);
ntrials = 150; %trials per cond
CorrSet = repmat(1:nconds,1,ntrials);
CorrSet = CorrSet(randperm(length(CorrSet)));
stims = {};
i = 1;

for i = 1:length(CorrSet)
    sprintf('Stim %d/%d',i,length(CorrSet))
    [stimABAB, envs, ERBspace, Tones_f] = Stim_Bind_AB1AB2_noise(Corr_inds{CorrSet(i),1}, ... 
        Corr_inds{CorrSet(i),2} , fs, f_start, f_end, Tones_num, ERB_spacing);
    stims{i} = stimABAB;
end

save('StimActive_AB1AB2noise.mat','stims','CorrSet','Corr_inds','fs')

ntrials = 10;
CorrSet_dem = repmat(1:nconds,1,ntrials);
CorrSet_dem =CorrSet_dem(randperm(length(CorrSet_dem)));
for j = 1:length(Corr_inds) + length(CorrSet_dem)
    sprintf('Demo %d/%d', j, length(CorrSet_dem) + length(Corr_inds))
    if j <=2
        [stimABAB, envs, ERBspace, Tones_f] = Stim_Bind_AB1AB2_noise(Corr_inds{j,1}, ... 
            Corr_inds{j,2} , fs, f_start, f_end, Tones_num, ERB_spacing);
    else
        [stimABAB, envs, ERBspace, Tones_f] = Stim_Bind_AB1AB2_noise(Corr_inds{CorrSet_dem(j-2),1}, ... 
            Corr_inds{CorrSet_dem(j-2),2} , fs, f_start, f_end, Tones_num, ERB_spacing);
    end
    stim_dem{j} = stimABAB;
end
save('StimActive_AB1AB2noise_dem.mat','stim_dem','CorrSet_dem','Corr_inds','fs')



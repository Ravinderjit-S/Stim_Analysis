clear all
load('Seed156.mat')
rng(s)
FMs_test = [2, 4, 8, 16, 32, 64, 128, 200, 240, 300];
ntrials = 10;
FMs_gen = repmat(FMs_test,1,ntrials);
stim_dur = 1; %duration of each noise, 3 will be played per trial
fs =48828.125;
BPfilt = designfilt('bandpassiir', 'StopbandFrequency1', 190, 'PassbandFrequency1', 200, 'PassbandFrequency2', 1500, 'StopbandFrequency2', 1550, 'StopbandAttenuation1', 30, 'PassbandRipple', 1, 'StopbandAttenuation2', 30, 'SampleRate', fs);

for i = 1:numel(FMs_gen)
    stims{i} = IACsinAFC3(stim_dur,fs,FMs_gen(i),BPfilt);
end

reorder = randperm(numel(FMs_gen));
FM_stim = FMs_gen(reorder);
stims = stims(reorder);
save('IACbehaviorStims.mat','FM_stim','stims')

    


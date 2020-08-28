clear

path = '../../CommonExperiment';
p = genpath(path);
addpath(p);

folder_loc = '/media/ravinderjit/Data_Drive/Data/Stimuli_WavMat/AMphi/';

stim_dur = 1.5;
frange = [500 6000];
fratio = 4;
dichotic = 0;
fs = 48828;
risetime = .125;

fm = 4;
phis = [30, 60, 90, 180];
ntrials = 10; 
phis = repmat(phis, 1, ntrials);
phis = phis(randperm(length(phis)));

for i = 1:length(phis)
    f1 = randi(frange(2)/fratio - frange(1)) + frange(1); 
    f2 = fratio*f1; 
    stim = SAM_phi(f1,f2,fs,stim_dur,fm,phis(i),dichotic);
    order = randperm(3);
    stim = stim(order);
    correct(i) = find(order ==3);
    for j = 1:3
        stimulus = stim{j};
        energy = mean(rms(stimulus'));
        stimulus(1,:) = rampsound(stimulus(1,:),fs,risetime) / energy;
        stimulus(2,:) = rampsound(stimulus(2,:),fs,risetime) / energy;
        stim{j} = stimulus;
    end
    stimulus_all = horzcat(stim{1}, zeros(2,fs*0.5), stim{2}, zeros(2,fs*0.5), stim{3});
    stimulus_all = scaleSound(stimulus_all);
    fname = [folder_loc 'AM_' num2str(fm) '_trial_' num2str(i) '.wav'];
    audiowrite(fname, stimulus_all',fs);
end
save([folder_loc 'StimData.mat'],'correct','fm','phis');

f1 = randi(frange(2)/fratio - frange(1)) + frange(1); 
f2 = fratio*f1; 
stim = SAM_phi(f1,f2,fs,stim_dur,fm,180,dichotic);
for j = 1:3
    stimulus = stim{j};
    energy = mean(rms(stimulus'));
    stimulus(1,:) = rampsound(stimulus(1,:),fs,risetime) / energy;
    stimulus(2,:) = rampsound(stimulus(2,:),fs,risetime) / energy;
    stim{j} = stimulus;
end
stimulus_all = horzcat(stim{1}, zeros(2,fs*0.750), stim{2}, zeros(2,fs*0.750), stim{3});
stimulus_all = scaleSound(stimulus_all);

for i = 1:5
    stimulus_all = horzcat(stimulus_all, zeros(2,fs*0.5), stimulus_all);
end
fname = [folder_loc 'volstim.wav'];
audiowrite(fname,stimulus_all',fs);




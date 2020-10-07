clear

path = '../../CommonExperiment';
p = genpath(path);
addpath(p);

path = '../Stim_Dev';
p = genpath(path);
addpath(p)

%folder_loc = '/media/ravinderjit/Data_Drive/Data/Stimuli_WavMat/AMphi/';
folder_loc = '/home/ravinderjit/Documents/OnlineStim_WavFiles/PhiAM/';

stim_dur = 1.5;
frange = [500 6000];
fratio = 4;
dichotic = 1;
ref = 1;
fs = 44100;
risetime = .125;

phi = 90;
fms = [4, 8, 16, 32, 64];
ntrials = 20; 
fms = repmat(fms, 1, ntrials);
fms = fms(randperm(length(fms)));
ISI = zeros(2,fs* (stim_dur/2)); 

for i = 1:length(fms)
    f1 = randi(frange(2)/fratio - frange(1)) + frange(1); 
    f2 = fratio*f1; 
    stim = SAM_phi(f1,f2,fs,stim_dur,fms(i),phi,dichotic, ref);
    order = [1 randperm(3)+1];
    stim = stim(order);
    correct(i) = find(order ==4);
    for j = 1:4
        stimulus = stim{j};
        energy = mean(rms(stimulus'));
        stimulus(1,:) = rampsound(stimulus(1,:),fs,risetime) / energy;
        stimulus(2,:) = rampsound(stimulus(2,:),fs,risetime) / energy;
        stim{j} = stimulus;
    end
    stimulus_all = horzcat(stim{1}, ISI, stim{2}, ISI, stim{3}, ISI, stim{4});
    stimulus_all = scaleSound(stimulus_all);
    fname = [folder_loc 'Phi_' num2str(phi) '_trial_' num2str(i) '.wav'];
    audiowrite(fname, stimulus_all',fs);
end
save([folder_loc 'StimData_' num2str(phi) '.mat'],'correct','fms','phi');
save(['StimData_' num2str(phi) '.mat'],'correct','fms','phi');

f1 = randi(frange(2)/fratio - frange(1)) + frange(1); 
f2 = fratio*f1; 
stim = SAM_phi(f1,f2,fs,stim_dur,16,90,dichotic,ref);
for j = 1:4
    stimulus = stim{j};
    energy = mean(rms(stimulus'));
    stimulus(1,:) = rampsound(stimulus(1,:),fs,risetime) / energy;
    stimulus(2,:) = rampsound(stimulus(2,:),fs,risetime) / energy;
    stim{j} = stimulus;
end
stimulus_all = horzcat(stim{1}, ISI, stim{2}, ISI, stim{3}, ISI, stim{4});
stimulus_all = scaleSound(stimulus_all);

for i = 1:3
    stimulus_all = horzcat(stimulus_all, ISI, stimulus_all);
end

fname = [folder_loc 'volstim.wav'];
audiowrite(fname,stimulus_all',fs);


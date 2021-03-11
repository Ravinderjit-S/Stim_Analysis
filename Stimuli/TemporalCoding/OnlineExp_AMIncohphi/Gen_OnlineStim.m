clear

path = '../../CommonExperiment';
p = genpath(path);
addpath(p);

path = '../Stim_Dev';
p = genpath(path);
addpath(p)

dichotic = 1;

%folder_loc = '/media/ravinderjit/Data_Drive/Data/Stimuli_WavMat/AMIncohphi/';
folder_loc = '/home/ravinderjit/Documents/OnlineStim_WavFiles/AMIncohphi/';

if dichotic
    folder_loc = '/home/ravinderjit/Documents/OnlineStim_WavFiles/AMIncohphi_diotic/'; 
end
    
stim_dur = 1.5;
frange = [500 6000];
fratio = 4;

ref = 1;
fs = 44100;
risetime = .125;

base_phi = 90;
fm = 64;
phis = [30, 60, 90];
ntrials = 20; 
phis = repmat(phis, 1, ntrials);
phis = phis(randperm(length(phis)));
ISI = zeros(2,fs* (stim_dur/2)); 

for i = 1:length(phis)
    f1 = randi(frange(2)/fratio - frange(1)) + frange(1); 
    f2 = fratio*f1; 
    stim = SAM_phi_incoh(f1,f2,fs,stim_dur,fm,base_phi,phis(i),dichotic,ref);
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
    if dichotic
        fname = [folder_loc 'AM_diotic' num2str(fm) '_trial_' num2str(i) '.wav'];
    else
        fname = [folder_loc 'AM_' num2str(fm) '_trial_' num2str(i) '.wav'];
    end
    audiowrite(fname, stimulus_all',fs);
end
if dichotic
    save([folder_loc 'StimData_diotic' num2str(fm) '.mat'],'correct','fm','phis'); 
    save(['StimData_diotic' num2str(fm) '.mat'],'correct','fm','phis');
else
    save([folder_loc 'StimData_' num2str(fm) '.mat'],'correct','fm','phis'); %#ok
    save(['StimData_' num2str(fm) '.mat'],'correct','fm','phis');
end

f1 = randi(frange(2)/fratio - frange(1)) + frange(1); 
f2 = fratio*f1; 
stim = SAM_phi_incoh(f1,f2,fs,stim_dur,fm,base_phi,90,dichotic,ref);
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

if dichotic
    fname = [folder_loc 'diotic_volstim.wav'];
else
    fname = [folder_loc 'volstim.wav'];
end
audiowrite(fname,stimulus_all',fs);




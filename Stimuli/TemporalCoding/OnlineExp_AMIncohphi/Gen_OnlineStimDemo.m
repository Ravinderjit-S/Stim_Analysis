clear

path = '../../CommonExperiment';
p = genpath(path);
addpath(p);

path = '../Stim_Dev';
p = genpath(path);
addpath(p);

dichotic = 1;


%folder_loc = '/media/ravinderjit/Data_Drive/Data/Stimuli_WavMat/AMphi/';
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

fm = 64;
base_phi =90;
phis = [30, 60, 90];
ntrials = 5;
phis = repmat(phis, 1, ntrials);
phis = phis(randperm(length(phis)));
ISI = zeros(2,fs* (stim_dur /2)); 

demo_phi = [90 * ones(1,5), 60 * ones(1,5), 30 * ones(1,5)];

for k = 1:length(demo_phi)
    f1 = randi(frange(2)/fratio - frange(1)) + frange(1); 
    f2 = fratio*f1; 
    stim = SAM_phi_incoh(f1,f2,fs,stim_dur,fm,base_phi,demo_phi(k),dichotic,ref);
    order = [1,2,3,4];
    if mod(k,5) ==4 || mod(k,5) ==0
        order = [1 randperm(3)+1];
    end
    stim = stim(order);
    correct_k(k) = find(order ==4);
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
        fname = [folder_loc 'DEMO_AMIncoh_diotic' num2str(fm) '_phi_' num2str(demo_phi(k)) '_trial_' num2str(k) '.wav'];
    else
        fname = [folder_loc 'DEMO_AMIncoh_diotic' num2str(fm) '_phi_' num2str(demo_phi(k)) '_trial_' num2str(k) '.wav'];
    end
    audiowrite(fname, stimulus_all',fs);
end
if dichotic
    save([folder_loc 'DEMOprac_StimData_Incohdiotic' num2str(fm) '.mat'],'correct_k','fm','phis');
    save(['DEMOprac_StimData_Incohdiotic' num2str(fm) '.mat'],'correct_k','fm','phis');
else
    save([folder_loc 'DEMOprac_StimData' num2str(fm) '.mat'],'correct_k','fm','phis');
    save(['DEMOprac_StimData_' num2str(fm) '.mat'],'correct_k','fm','phis');
end

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
        fname = [folder_loc 'DEMO_AMIncoh_diotic' num2str(fm) '_trial_' num2str(i) '.wav'];
    else
        fname = [folder_loc 'DEMO_AMIncoh_' num2str(fm) '_trial_' num2str(i) '.wav'];
    end
    audiowrite(fname, stimulus_all',fs);
end
if dichotic
    save([folder_loc 'DEMO_StimData_diotic' num2str(fm) '.mat'],'correct','fm','phis');
    save(['DEMO_StimData_diotic' num2str(fm) '.mat'],'correct','fm','phis');
else
    save([folder_loc 'DEMO_StimData_' num2str(fm) '.mat'],'correct','fm','phis');
    save(['DEMO_StimData_' num2str(fm) '.mat'],'correct','fm','phis');
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
    fname = [folder_loc 'DEMO_volstim.wav'];
else
    fname = [folder_loc 'DEMO_volstim.wav'];
end
audiowrite(fname,stimulus_all',fs);





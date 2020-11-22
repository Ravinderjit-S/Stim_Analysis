% Making Stims for EEG expereiment

clear
clc
path = '../CommonExperiment';
addpath(genpath(path));
addpath('StimDev')

fs = 48828;
center_f = 4000;
ERB_halfwidth = 0.5;
ERBspacing = 1.5;
noise_bands = CMRbands(center_f, ERB_halfwidth, ERBspacing);
mod_band = [2,10];
mod_bpfo = 1/2*5*fs;
t_coh = 1;
t_incoh = 1;
Total_t = 2*t_coh+2*t_incoh;

Stim_10 = zeros(300, Total_t* fs);
mods_10 = cell(1,300);
for i =1:300
    fprintf('Sec 1 Trial %d/300 \n',i);
    [Sig,mods] = Noise_coherenceABAB(noise_bands,mod_band,mod_bpfo,t_coh,t_incoh,fs);
    Stim_10(i,:) = Sig;
    mods_10{i} = mods;
end

mod_band = [36 44];
Stim_44 = zeros(300,Total_t*fs);
mods_44 = cell(1,300);
for i =1:300
    fprintf('Sec 2 Trial %d/300 \n',i);
    [Sig,mods] = Noise_coherenceABAB(noise_bands,mod_band,mod_bpfo,t_coh,t_incoh,fs);
    Stim_44(i,:) = Sig;
    mods_44{i} = mods;
end

mod_band = [101 109];
Stim_109 = zeros(300,Total_t*fs);
mods_109 = cell(1,300);
for i =1:300
    fprintf('Sec 2 Trial %d/300 \n',i);
    [Sig,mods] = Noise_coherenceABAB(noise_bands,mod_band,mod_bpfo,t_coh,t_incoh,fs);
    Stim_109(i,:) = Sig;
    mods_109{i} = mods;
end

save('Noise_coherenceABAB_EEGstims.mat','Stim_10','Stim_44','Stim_109','mods_10','mods_44','mods_109')







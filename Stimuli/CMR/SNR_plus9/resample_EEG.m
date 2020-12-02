clear
load('CMRrandmod_2_10_coh_1.mat')

for i =1:4
    target_mods_EEG(i,:) = resample(target_mods(i,:),4096,48828);
end

save('target_mods_9dB.mat','target_mods_EEG')


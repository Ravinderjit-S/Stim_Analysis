% Make EEG stims for CMRrandMod experiment


clear
path = '../CommonExperiment';
addpath(genpath(path));
addpath('StimDev')


fs = 48828;
tlen = 4;
t = 0:1/fs:tlen-1/fs;

ERB_halfwidth = 0.5;
ERBspacing = 1.5;
target_f = 4000;
noise_bands = CMRbands(target_f, ERB_halfwidth, ERBspacing);

SNRdb = 12;
mod_band = [2 10];

%target_modf = [4, 40, 223];
target_modf = [4,40];
target_modf = 223
target_mod_band = [2 10];

coh = [0 1];
bp_mod_fo = 1/2 * 5 *fs; %filter order for slowest modulation instance ... keep same sharpness for all modulation filters so setting here

trials = 300;
target_mods = zeros(length(target_modf),length(t));
for i =1:length(target_modf)
    target_mods(i,:) = sin(2*pi*target_modf(i).*t);
end
target_mods = max(target_mods,0);

% bp_filt_mod = fir1(bp_mod_fo, [target_mod_band(1) target_mod_band(2)]*2/fs,'bandpass');
% 
% noise_mod = randn(1,1.5*length(t) + bp_mod_fo + 1);
% noise_mod = filter(bp_filt_mod,1,noise_mod);
% noise_mod = noise_mod(bp_mod_fo+1:bp_mod_fo+length(t));
% noise_mod = noise_mod - min(noise_mod);
% noise_mod = noise_mod / max(noise_mod);
% 
% target_mods(end+1,:) = noise_mod;

for k = 1:length(coh)
    This_coh = coh(k);
    for i =1:length(target_modf)
        Sig = zeros(trials,length(t));  
        for j = 1:trials
            fprintf('BigBlock %d/%d Stim block %d/%d  Stim %d/%d \n',k,length(coh),i,length(target_modf),j,trials)
            Sig(j,:) = CMR_randMod(noise_bands,target_f,SNRdb,mod_band,target_mods(i,:),fs,tlen,coh(k),bp_mod_fo);
        end
        if i == length(target_modf)+1
            save(['CMRrandmod_tpsd_' num2str(target_mod_band(1)) '_' num2str(target_mod_band(2)) '_coh_' num2str(coh(k)) '.mat'],'Sig','This_coh','fs','SNRdb','mod_band','target_mod_band','target_mods') 
        else
            save(['CMRrandmod_tpsd_' num2str(target_modf(i)) '_coh_' num2str(coh(k)) '.mat'],'Sig','This_coh','fs','SNRdb','mod_band') 
        end
    end
end






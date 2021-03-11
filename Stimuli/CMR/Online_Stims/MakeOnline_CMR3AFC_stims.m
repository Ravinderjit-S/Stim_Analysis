clear

path = '../../CommonExperiment';
p = genpath(path);
addpath(p);

addpath('../StimDev')

fs = 44100;
tlen = 1;
t = 0:1/fs:tlen-1/fs;
risetime = .050; 

ERB_halfwidth = 0.5;
ERBspacing = 1.5;
target_f = 4000;
noise_bands = CMRbands(target_f, ERB_halfwidth, ERBspacing);
SNRdB_1 = 4:-6:-38;
SNRdB_2 = 6:-6:-30; 
mod_bands = [2 10; 36 44; 131 139];
mod_bands = [16 24];
%mod_bands = [12 20; 20 50];
target_modf = 0;
bp_mod_fo = 1/2 * 5 *fs;
ntrials = 7;

silence = zeros(2,round(fs)*0.5);

folder_loc = '/home/ravinderjit/Documents/OnlineStim_WavFiles/CMR/';
folder_loc = '/media/ravinderjit/Data_Drive/Data/Stimuli_WavMat/CMR_frozenMod/';


for k = 1:length(mod_bands)
    if k ==1
        SNRdB_exp = repmat([[SNRdB_1 SNRdB_2]; [ones(1,length(SNRdB_1)) zeros(1,length(SNRdB_2))]],1,ntrials)';
    else
        SNRdB_exp = repmat([[SNRdB_2 SNRdB_2]; [ones(1,length(SNRdB_2)) zeros(1,length(SNRdB_2))]],1,ntrials)';
    end
    exp_modband = mod_bands(k,:);
    correct = [];
    for l =1:size(SNRdB_exp,1)
        fprintf('Band: %d/%d  Stim: %d/%d \n',k,length(mod_bands),l,size(SNRdB_exp,1))
        stim = CMR_randMod_3AFC(noise_bands,target_f,SNRdB_exp(l,1),exp_modband,target_modf,fs,tlen,SNRdB_exp(l,2),bp_mod_fo);
        order = randperm(3);
        stim = stim(order);
        correct(l) = find(order==3);
        for m =1:3
            stimulus = stim{m};
            energy = mean(rms(stimulus));
            stimulus = rampsound(stimulus,fs,risetime) / energy;
            stimulus(2,:) = stimulus;
            stim{m} = scaleSound(stimulus);
        end
        stimulus_all = horzcat(stim{1}, silence, stim{2}, silence, stim{3});
        Fold_path = [folder_loc 'Mod_' num2str(exp_modband(1)) '_' num2str(exp_modband(2)) '/' ];
        fname = ['CMR3AFCrandMod_' num2str(exp_modband(1)) '_' num2str(exp_modband(2)) '_trial_' num2str(l) '.wav'];
        audiowrite([Fold_path fname],stimulus_all',fs);
    end
    save([Fold_path 'Stim_Data.mat'],'correct','exp_modband','SNRdB_exp')
    stimulus_vol = horzcat(stimulus_all, silence, stimulus_all, silence, stimulus_all, silence, stimulus_all);
    audiowrite([Fold_path 'volstim.wav'],stimulus_vol',fs);
end

            
        
        
   





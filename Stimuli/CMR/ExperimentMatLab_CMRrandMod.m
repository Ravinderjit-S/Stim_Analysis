%2 down 1 up experiment in Matlab
clear
clc
path = '../CommonExperiment';
addpath(genpath(path));
addpath('StimDev')

fs = 44100;
tlen = 1;
t = 0:1/fs:tlen-1/fs;

ERB_halfwidth = 0.5;
ERBspacing = 1.5;
target_f = 4000;
noise_bands = CMRbands(target_f, ERB_halfwidth, ERBspacing);

mod_band = [32 40];
target_modf = 0;
coh = 1;
bp_mod_fo = 1/2 * 5 *fs;

Reversals = 0;
Reversals_stop = 11;
Reversals_changeRes = 4;
Start_SNR = 0;
SNR_i = Start_SNR;
StartResolution = 7;
EndResolution = 2;
Correct_2up = [0 0];
TrackSNRs = Start_SNR;
changes =0;

n_mods = load(['RandMod_' num2str(mod_band(2)) '.mat']);
n_mods_iter = 1;
while Reversals < Reversals_stop
    if coh == 1
        noise_mod = n_mods.noise_mod(n_mods_iter,:);
        n_mods_iter = n_mods_iter+1;
    else
        noise_mod = n_mods.noise_mod(n_mods_iter:n_mods_iter+2,:);
        n_mods_iter = n_mods_iter+3;
    end
    [stim] = CMR_randMod_giveNoiseMod(noise_bands,target_f,SNR_i,mod_band,target_modf,fs,tlen,noise_mod);
    
    if Reversals < Reversals_changeRes
        ChangeSNR = StartResolution;
    else
        ChangeSNR = EndResolution;
    end
    
    PlayOrder = randperm(3);
    stim = stim(PlayOrder);
    
    for j = 1:3
        x = stim{j};
        x = rampsound(x,fs,.050);
        x = scaleSound(x);
        soundsc(x,fs); 
        pause(tlen+0.5*tlen);
    end
    resp = input('Anwer: ');
    answer = find(PlayOrder==3);
    Correct = resp == answer;
    Correct_2up = circshift(Correct_2up,1); Correct_2up(1) = Correct;

    if Correct 
        if all(Correct_2up)
            SNR_i = SNR_i - ChangeSNR;
            Correct_2up = [0 0];
        end
    else
        SNR_i = SNR_i + ChangeSNR; 
    end
    
    TrackSNRs = [TrackSNRs SNR_i];%#ok
    CurLength = length(changes);
    changes = sign(diff(TrackSNRs)); changes= changes(abs(changes)>0);
    if length(changes)>1 && changes(end) ~=changes(end-1) && length(changes) > CurLength
        Reversals = Reversals + 1;
    end
end
save(['CMR_SigAM_' num2str(mod_band(2)) 'coh_' num2str(coh) '.mat'],'TrackSNRs','target_modf','target_f','noise_bands','coh','mod_band');
    


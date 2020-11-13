%2 down 1 up experiment in Matlab
clear
clc
path = '../CommonExperiment';
addpath(genpath(path));

fs = 44100;
tlen = 1;
t = 0:1/fs:tlen-1/fs;
upper_f = 4000;
target_f = 2000;
lower_f = 1000;
noise_half_bw = 300; %half the bandwidth of noise. noise is centered on lower and upper f
center_freqs = [lower_f target_f upper_f];
n_modf = 10;
target_modf = 223;
phase = 180;

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

while Reversals < Reversals_stop
    [stim] = CMR(center_freqs,noise_half_bw,SNR_i,n_modf,target_modf,fs,tlen,phase);
    
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
save(['CMR_SigAM_' num2str(target_modf) 'phase_' num2str(phase) '.mat'],'TrackSNRs','target_modf','center_freqs','noise_half_bw','phase');
    


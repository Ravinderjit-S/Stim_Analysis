function [stim, noiseRMS,total_dur] = ITDt_ToneDetect(ToneDur, fs, BPfilt,fm,SNRdes,StaticIAC)
%Da is input as a column
%stim is 3AFC with the 3rd one being correct
%SNR in dB 
%StaticIAC should be empty, 1, or 0 for this experiment


fc = 800;


Pre_Post_dur = 0.2;
total_dur = Pre_Post_dur*2 + ToneDur;
t = 0:1/fs:ToneDur-1/fs;

nbn1 = randn(2, round(total_dur*3*fs)); %made noise longer to deal with filter transients

if isempty(StaticIAC)
    ToneL = sin(2*pi*(fc-fm).*t);
    ToneR = sin(2*pi*(fc+fm).*t);
    BinBeat = [ToneL;ToneR];
else
    if ~any(StaticIAC== [-1 1])
        error('Check StaticIAC value')
    end
    if StaticIAC==1
        ToneL = sin(2*pi*fc.*t);
        BinBeat = [ToneL;ToneL]; 
    end
    if StaticIAC==-1
        ToneL = sin(2*pi*fc.*t);
        BinBeat = [ToneL; -ToneL];
    end
end
BinBeat(1,:) = rampsound(BinBeat(1,:),fs,.010);
BinBeat(2,:) = rampsound(BinBeat(2,:),fs,.010);

lenNBN = round(total_dur*fs);

nbn1 = filter(BPfilt, nbn1');
nbn1 = nbn1(lenNBN+1:2*lenNBN,:)';

BinBeat_sil = horzcat(zeros(2,floor(Pre_Post_dur*fs)), BinBeat, zeros(2,ceil(Pre_Post_dur*fs)));
Da_on_index = floor(Pre_Post_dur*fs)+1:length(BinBeat_sil)-ceil(Pre_Post_dur*fs);

alpha1 = 10^(SNRdes/20)*rms(nbn1(1,Da_on_index))./rms(BinBeat_sil(1,Da_on_index));
alpha2 = 10^(SNRdes/20)*rms(nbn1(2,Da_on_index))./rms(BinBeat_sil(2,Da_on_index));

BinBeat_sil(1,:) = alpha1.*BinBeat_sil(1,:); BinBeat_sil(2,:) = alpha2.*BinBeat_sil(2,:);

stim_BinBeat = nbn1 + BinBeat_sil;
noiseRMS(1) = rms(nbn1(1,:));
noiseRMS(2) = rms(nbn1(2,:));

stim = [{nbn1}, {nbn1}, {stim_BinBeat}];

end


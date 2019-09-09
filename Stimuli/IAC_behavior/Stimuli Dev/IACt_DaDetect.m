function [stim, noiseRMS,total_dur] = IACt_DaDetect(Da, fs, BPfilt,fm,SNRdes,StaticIAC)
%Da is input as a column
%stim is 3AFC with the 3rd one being correct
%SNR in dB 
%StaticIAC should be empty, 1, or 0 for this experiment

% Da_len = length(Da);
% Da = filtfilt(BPfilt, vertcat(zeros(size(Da)),Da,zeros(size(Da))));
% Da = Da(Da_len+1:2*Da_len);

Da = rampsound(Da,fs,0.010);

%Da = vertcat(Da,Da,Da,Da);
Da = vertcat(Da',-Da');
Da_dur = length(Da)/fs;
Pre_Post_dur = 0.2;
total_dur = Pre_Post_dur*2 + Da_dur;

nbn1 = randn(2, round(total_dur*3*fs)); %made noise longer to deal with filter transients

t = 0:1/fs:size(nbn1,2)/fs-1/fs;
A = sin(2*pi*fm.*t);
B = sqrt(1-A.^2);

if isempty(StaticIAC)
    nbn1(2,:) = A.*nbn1(1,:) + B.*nbn1(2,:);
else
    if ~any(StaticIAC== [0 1])
        error('Check StaticIAC value')
    end
    if StaticIAC==1
        nbn1(2,:) = StaticIAC.*(nbn1(1,:));
    end
end

lenNBN = round(total_dur*fs);

nbn1 = filter(BPfilt, nbn1');
nbn1 = nbn1(lenNBN+1:2*lenNBN,:)';

Da_sil = horzcat(zeros(2,floor(Pre_Post_dur*fs)), Da, zeros(2,ceil(Pre_Post_dur*fs)));
Da_on_index = floor(Pre_Post_dur*fs)+1:length(Da_sil)-ceil(Pre_Post_dur*fs);

alpha1 = 10^(SNRdes/20)*rms(nbn1(1,Da_on_index))./rms(Da_sil(1,Da_on_index));
alpha2 = 10^(SNRdes/20)*rms(nbn1(2,Da_on_index))./rms(Da_sil(2,Da_on_index));

Da_sil(1,:) = alpha1.*Da_sil(1,:); Da_sil(2,:) = alpha2.*Da_sil(2,:);

stim_Da = nbn1 + Da_sil;
noiseRMS(1) = rms(nbn1(1,:));
noiseRMS(2) = rms(nbn1(2,:));

stim = [{nbn1}, {nbn1}, {stim_Da}];

end


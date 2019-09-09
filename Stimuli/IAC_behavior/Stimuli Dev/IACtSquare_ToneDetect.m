function [stim, noiseRMS,total_dur] = IACtSquare_ToneDetect(fs, BPfilt,WindowSize,SNRdes,StaticIAC)

%stim is 3AFC with the 3rd one being correct
%SNR in dB 
%StaticIAC should be empty, 1, or 0 for this experiment
%window size in seconds

if isempty(StaticIAC)
    total_dur = 0.8+WindowSize; % 400 ms of uncorrelated noise on both sides with correlated noise in middle
else
    total_dur = 0.8;
end
tone_dur = 0.6; %10 ms
t = 0:1/fs:tone_dur;
tone = sin(2*pi*850*t); 
tone = rampsound(tone,fs,.005); 
tone = [tone;-tone]; % tone has IAC=-1

nbn1 = randn(2, round(total_dur*2.2*fs)); %made noise longer to deal with filter transients
lenNBN = round(total_dur*fs);

if isempty(StaticIAC)
    Window_midInd = round(total_dur*0.5*fs); 
    Window_on_ind = Window_midInd - round(fs*WindowSize/2):Window_midInd + round(fs*WindowSize/2);
    nbn1(2,Window_on_ind+lenNBN+1) = nbn1(1,Window_on_ind+lenNBN+1); %correlated noise in middle window
    nbn1 = filtfilt(BPfilt, nbn1');
    nbn1 = nbn1(lenNBN+1:2*lenNBN,:)';
elseif StaticIAC == 1
    nbn1 = filter(BPfilt,nbn1');
    nbn1 = nbn1(lenNBN+1:2*lenNBN,:)';
    nbn1(2,:) = nbn1(1,:);
elseif StaticIAC == 0
    nbn1 = filter(BPfilt, nbn1');
    nbn1 = nbn1(lenNBN+1:2*lenNBN,:)';
end



Tone_midInd = round(total_dur*0.5*fs);
Tone_on_inds = Tone_midInd - round(fs*tone_dur/2):Tone_midInd + round(tone_dur/2*fs); 


alpha1 = 10^(SNRdes/20)*rms(nbn1(1,:))./rms(tone(1,:));
alpha2 = 10^(SNRdes/20)*rms(nbn1(2,:))./rms(tone(2,:));

alpha = (alpha1+alpha2)./2;

tone(1,:) = alpha.*tone(1,:); tone(2,:) = alpha.*tone(2,:);

stim_nbnTone = nbn1;
stim_nbnTone(:,Tone_on_inds) = nbn1(:,Tone_on_inds) + tone;
noiseRMS(1) = rms(nbn1(1,:));
noiseRMS(2) = rms(nbn1(2,:));

stim = [{nbn1}, {nbn1}, {stim_nbnTone}];
 
end


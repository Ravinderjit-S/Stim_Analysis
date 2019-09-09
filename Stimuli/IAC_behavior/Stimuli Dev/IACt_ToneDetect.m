function [stim, noiseRMS,total_dur] = IACt_ToneDetect(fs, BPfilt,fm,SNRdes,StaticIAC)
%Da is input as a column
%stim is 3AFC with the 3rd one being correct
%SNR in dB 
%StaticIAC should be empty, 1, or 0 for this experiment


total_dur = 1.0; 
tone_dur = .02; 
t = 0:1/fs:tone_dur;
tone = sin(2*pi*850*t); 
tone = rampsound(tone,fs,.004); 
tone = [tone;-tone]; % tone has IAC=-1

nbn1 = randn(2, round(total_dur*3*fs)); %made noise longer to deal with filter transients

t = 0:1/fs:size(nbn1,2)/fs-1/fs;
A = sin(2*pi*fm.*t);
B = sqrt(1-A.^2);

if isempty(StaticIAC)
    nbn1(2,:) = A.*nbn1(1,:) + B.*nbn1(2,:);
    IAC_peaks = find(A>0.9999); % indexes that IAC is 1
    Tone_midInd = IAC_peaks(find(t(IAC_peaks) > 0.5,1,'first')); % index for half way thru playing tone
    Tone_on_inds = Tone_midInd - round(fs*tone_dur/2):Tone_midInd + round(tone_dur/2*fs); 
else
    if ~any(StaticIAC== [0 1])
        error('Check StaticIAC value')
    end
    if StaticIAC==1
        nbn1(2,:) = StaticIAC.*(nbn1(1,:));
    end
    Tone_midInd = round(0.5*fs);
    Tone_on_inds = Tone_midInd - round(fs*tone_dur/2):Tone_midInd + round(tone_dur/2*fs); 
end

lenNBN = round(total_dur*fs);

nbn1 = filter(BPfilt, nbn1');
nbn1 = nbn1(lenNBN+1:2*lenNBN,:)';


alpha1 = 10^(SNRdes/20)*rms(nbn1(1,Tone_on_inds))./rms(tone(1,:));
alpha2 = alpha1;
% alpha2 = 10^(SNRdes/20)*rms(nbn1(2,Tone_on_inds))./rms(tone(2,:));

tone(1,:) = alpha1.*tone(1,:); tone(2,:) = alpha2.*tone(2,:);

stim_nbnTone = nbn1;
stim_nbnTone(:,Tone_on_inds) = nbn1(:,Tone_on_inds) + tone;
noiseRMS(1) = rms(nbn1(1,:));
noiseRMS(2) = rms(nbn1(2,:));

stim = [{nbn1}, {nbn1}, {stim_nbnTone}];

figure,plot(t,A) 
hold on, plot(t(Tone_on_inds),A(Tone_on_inds),'rx'), hold off

end


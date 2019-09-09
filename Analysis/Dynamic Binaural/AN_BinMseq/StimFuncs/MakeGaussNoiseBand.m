function [LWaveform,RWaveform] = ...
   MakeGaussNoiseBand(fLow, fHigh, dur, Fsample, Chan, Rho, ITD, IPD, RandSeed,ContraChan)
% MAKEGAUSSNOISEBAND - generates gaussian bandpass noise
% inputs:
% fLow: lowest frequency in passband (in Hz)
% fHigh: highest frequency in passband (in Hz)
% dur: duration (in milliseconds)
% Fsample: sampling frequency (in Hz)
% Chan: number of channels (1, or 2)
% Rho: inter-aural correlation ([-1,1])
% ITD: inter-aural time delay (in milliseconds)
% IPD: inter-aural phase delay (in cycles)
% RandSeed: can either specify a seed (e.g., 123), or an empty vector and
% it will pick some random number.
% ContraChan: which channel is contralateral (1 = left, 2 = right);
%Output:
%Waveform: self-explanatory
NeedTwoWaveForms_Rho = abs(Rho)~=1; % mixing is called for
Need_ITD_IPD = abs(ITD)~=0 || abs(IPD)~=0; %phase or time shift required
if isempty(ContraChan)
    ContraChan = 1;
    disp 'Default: Left channel assigned to contralateral ear';
end
% # spectral points upto Nyquist freq; round to next power of two to increase computational  speed
Nsamples = floor(dur*1e-3*Fsample);
NsamF = 2^nextpow2(0.5*Nsamples);
NsamT = 2*NsamF;
df = Fsample/NsamT; % freq spacing
freq = linspace(0,Fsample-df,NsamT)'; % freq axis
iLow = max(1,round(0.5+fLow/Fsample*NsamT)); % avoid negative indices
iHigh = round(0.5+fHigh/Fsample*NsamT);
NheadZero = iLow-1; % # zero-valued spectral points at low-freq side
NnonZero = max(1,iHigh-iLow+1); % convention: zero BW -> single tone
% # zero-valued spectral points at high-freq side (upto Nyquist):
NtrailZero = NsamF-NheadZero-NnonZero;
% compute passband portion of spectrum
SetRandState(RandSeed);
plainSpec = randn(NnonZero,1)+ 1i*randn(NnonZero,1);
if NeedTwoWaveForms_Rho
   MixplainSpec = randn(NnonZero,1)+ 1i*randn(NnonZero,1);
end
if isequal(Chan,2)
    if NeedTwoWaveForms_Rho
        % mix noises so as to obtain correct rho (see MvdH & Trahiotis, 1997)
        rPlus = ((1+Rho)/2)^0.5;
        rMin =  ((1-Rho)/2 )^0.5;
        plainSpecLeft =  (rPlus*plainSpec + rMin*MixplainSpec);
        plainSpecRight = (rPlus*plainSpec - rMin*MixplainSpec);
    else % trivial rho = +-1
        plainSpecLeft =  plainSpec;
        plainSpecRight =  Rho*plainSpec;
    end
    if Need_ITD_IPD%Apply an inter-aural time or phase delay
        p = (2*pi)*(IPD + (freq(iLow:iHigh)*(ITD/1000)));
        if ContraChan==1%i.e. ipsi is right, contra is left
            plainSpecLeft = plainSpecLeft.*(exp(1i*p));
        elseif ContraChan==2
            plainSpecRight = plainSpecRight.*(exp(1i*p));
        end
    end
    WaveformLeft = real(ifft([zeros(NheadZero,1); plainSpecLeft; zeros(NtrailZero,1); zeros(NsamF,1)]));
    WaveformRight = real(ifft([zeros(NheadZero,1); plainSpecRight; zeros(NtrailZero,1); zeros(NsamF,1)]));
    Waveform = [WaveformLeft WaveformRight];
else
   Waveform = real(ifft([zeros(NheadZero,1); plainSpec; zeros(NtrailZero,1); zeros(NsamF,1)]));
end
Waveform = Waveform(1:Nsamples,:);
%Scale waveform to make the rms = 1/(2^3) Volts. This means that absolute
%values greater than 1 are more than 3 sigma from the (zero) mean.
ScaleF = 0.1250./rms(Waveform);
Waveform = ScaleF.*Waveform;
LWaveform = Waveform(:,1);
if isequal(Chan,2)
    RWaveform = Waveform(:,2);
else
    RWaveform = [];
end
return;
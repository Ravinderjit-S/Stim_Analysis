function x = makeCMRstim_mod(fc, fs, fm, ofmbw, ofSNR, flankdist, flankbw,...
    condition, dur, ramp, target_modf)

% USAGE:
%   x = makeCMRstim(fc, fs, fm, ofmbw, ofSNR, flankdist, flankbw,...
%    condition, dur, ramp);
% INPUTS:
%   fc: Center frequency of target (Hz)
%   fs: Sampling rate (Hz)
%   fm: Masker modulation frequency (Hz)
%   ofmbw: On-frequency masker bandwidth (in ERB units)
%   ofSNR: On-frequency SNR (dB RMS)
%   flankdist: How far are flankers away from target (in ERB units)
%   flankbw: Flanker bandwidth (in ERB units)
%   condition: comodulated (1), codeviant (2), or absent/noflanker (0)
%   dur: Duration of stimulus (s)
%   ramp: Duration of ramp on either end (s)
%
% OUTPUTS:
%   x: Generated stimulus (1 x samples)
%
% NOTE:
%   Flanker and OFM are same RMS level of 1.0
%--------------------------------------
% Copyright Hari Bharadwaj. All Rights Reserved
%--------------------------------------

t = 0:(1/fs):(dur - 1/fs);

%% Make target
target_mod = max(sin(2*pi*target_modf*t),0);
%target_mod = 0.5+0.5*sin(2*pi*target_modf*t);
target = rampsound(target_mod.*sin(2*pi*fc*t), fs, ramp);
target = target / rms(target);

%% Make on-frequency masker
env = rampsound(0.5 + 0.5 * sin(2*pi*fm*t), fs, ramp);
env_dev = rampsound(0.5 - 0.5 * sin(2*pi*fm*t), fs, ramp);
% ERB to Hz conversion
bw = invcams(cams(fc) + ofmbw/2) - invcams(cams(fc) - ofmbw/2); 
ofm = makeNBNoiseFFT(bw, fc, dur, fs, ramp, 0)' .* env;
ofm = ofm / rms(ofm);

%% Make lowside flanker
fend = invcams(cams(fc) - flankdist);
fstart = invcams(cams(fc) - flankdist - flankbw);
fmid = (fend + fstart)/2;
bw = fend - fstart;
flank_low = makeNBNoiseFFT(bw, fmid, dur, fs, ramp, 0)';


%% Make highside flanker
fstart = invcams(cams(fc) + flankdist);
fend = invcams(cams(fc) + flankdist + flankbw);
fmid = (fend + fstart)/2;
bw = fend - fstart;
flank_high = makeNBNoiseFFT(bw, fmid, dur, fs, ramp, 0)';


%% Make overall stim
switch condition
    case 0
        x = ofm + db2mag(ofSNR)*target;
    case 1
        flank_low = env.*flank_low;
        flank_low = flank_low / rms(flank_low);
        flank_high = env.*flank_high;
        flank_high = flank_high / rms(flank_high);
        x = ofm + db2mag(ofSNR)*target + flank_low + flank_high;
    case 2
        flank_low = env_dev.*flank_low;
        flank_low = flank_low / rms(flank_low);
        flank_high = env_dev.*flank_high;
        flank_high = flank_high / rms(flank_high);
        x = ofm + db2mag(ofSNR)*target + flank_low + flank_high;
end



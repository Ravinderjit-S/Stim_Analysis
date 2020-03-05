function s = phoneSens_ER2(freq)
% Sensitivity of the ER-2
% Units are in dB SPL/1Vrms
% USAGE:
%   s = phoneSens(freq);
%
% freq - Frequency (scalar or vector) in Hz.
% s - Sensitivity at frequency 'freq' in dB SPL per 0 dBVrms
if any(freq < 63) || any(freq > 16000)
    error('Only frequencies between 63 Hz and 16 kHz are valid!');
end
s = 100;
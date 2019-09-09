function s = phoneSens(freq)
% Sensitivity of the HDA 300 headphone on BK 4153 with cheekplate
% Units are in dB SPL/1Vrms
% If sensitivity not originally measured at a certain frequency, it is
% interpolated and returned.
% USAGE:
%   s = phoneSens(freq);
%
% freq - Frequency (scalar or vector) in Hz.
% s - Sensitivity at frequency 'freq' in dB SPL per 0 dBVrms
if any(freq < 63) || any(freq > 16000)
    error('Only frequencies between 63 Hz and 16 kHz are valid!');
end
% Original measured frequencies
f = [63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 750, 800,...
    1000, 1250, 1500, 1600, 2000, 2500, 3000, 3150, 4000, 5000, 6300,...
    7500, 8000, 10000, 12000, 12500, 14000, 16000];

% Sensitivity in dB re: 1Pa/V
S = [34.9, 35.6, 36.1, 36.1, 37.2, 38, 38, 38.6, 38.2, 37.2, 35.4,...
    33.5, 32.7, 30.1, 27.5, 25.2, 24.3, 21.1, 20.3, 20, 19.8, 16.3, ...
    26.1, 26.4, 26.7, 28.1, 19.4, 24.1, 20.2, 17.5, 16.1];

Pref = 20e-6;
s = interp1(f, S, freq, 'spline') + db(1 / Pref);
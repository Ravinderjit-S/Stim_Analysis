function [nbn] = makeNBNfft_binaural(flow,fhigh,dur,fs,rho,ITD)
%flow,fhigh, and fs in Hz
%dur in secs

%implement later maybe: adjust samples via pow2 to increase computational speed ...

if flow <0 || fhigh > fs/2
    error('flow || fhigh is not cool')
end
if abs(rho) ~=1
    error('Currently rho can only be +1 or -1')
end
if rho ==-1 & ITD~=0
    error('These input parameters may currently be an issue')
end

dur_og = dur;
extend = 1; %a value of 1 leads to doubling the signal
dur = dur*(1+extend); %to deal with edge effects making signal longer and then gonna chop off edges
fstep = 1/dur; 
i_flow = round(flow*dur);
i_fhigh = round(fhigh*dur);
fullspec = zeros(1,round(dur*fs));

mag = ones(1,i_fhigh-i_flow+1); 
phase =  1i*rand(1,i_fhigh-i_flow+1)*(2*pi); %1i*randn(1,i_fhigh-i_flow+1);
fullspec(i_flow:i_fhigh) = mag+phase; fullspec2 = fullspec; %e^jtheta = cos(theta) + jsin(theta)
p = (2*pi) *ITD*(flow:fstep:fhigh); %how much to adjust phase to appply ITD
fullspec2(i_flow:i_fhigh) = fullspec(i_flow:i_fhigh).*(exp(1i*p)); %This line adds the phases calculated in p to the phase of the noise

fullspec = [conj(fullspec) fullspec];
fullspec2 = [conj(fullspec2) fullspec2];

nbn = ifft(fullspec,'symmetric');
nbn2 = rho*ifft(fullspec2,'symmetric');

nbn = nbn(round(extend/2*dur_og*fs+1):end-round(extend/2*dur_og*fs));
nbn2 = nbn2(round(extend/2*dur_og*fs+1):end-round(extend/2*dur_og*fs));%chopping off extra time to deal with edge effects

nbn = [nbn;nbn2];


end
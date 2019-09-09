function [nbn] = makeNBNfft_binaural_V3(flow,fhigh,dur,fs,rho,ITD,IPD)
%flow,fhigh, and fs in Hz
%dur in secs
%IPD in radians, ITD in secs

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
extend = 0; %a value of 1 leads to doubling the signal
dur = dur*(1+extend); %to deal with edge effects making signal longer and then gonna chop off edges
fstep = 1/dur; 
ind_flow = round(flow/fstep)+1;
ind_fhigh = round(fhigh/fstep)+1;
fullspec = zeros(1,round(dur*fs));

phase =  rand(1,ind_fhigh-ind_flow+1)*(2*pi); %1i*randn(1,i_fhigh-i_flow+1);
fullspec(ind_flow:ind_fhigh) = exp(1i*phase); %mag * exp(i*phase)
fullspec2 = fullspec; %e^jtheta = cos(theta) + jsin(theta)

p_ITD = (2*pi) *ITD*(flow:fstep:fhigh); %how much to adjust phase to appply ITD
fullspec2(ind_flow:ind_fhigh) = fullspec(ind_flow:ind_fhigh).*(exp(1j*p_ITD)); %This line adds the phases calculated in p to the phase of the noise
fullspec2(ind_flow:ind_fhigh) = fullspec2(ind_flow:ind_fhigh).*(exp(1j*IPD)); %This line adds the IPD to phase of noise

fullspec(end-ind_fhigh+1:end-ind_flow+1) = conj(fliplr(fullspec(ind_flow:ind_fhigh)));
fullspec2(end-ind_fhigh+1:end-ind_flow+1) = conj(fliplr(fullspec2(ind_flow:ind_fhigh)));

nbn = ifft(fullspec,'symmetric');
nbn2 = rho*ifft(fullspec2,'symmetric');

if extend~=0
    nbn = nbn(round(extend/2*dur_og*fs+1):end-round(extend/2*dur_og*fs));
    nbn2 = nbn2(round(extend/2*dur_og*fs+1):end-round(extend/2*dur_og*fs));%chopping off extra time to deal with edge effects
end
nbn = [nbn;nbn2];


end
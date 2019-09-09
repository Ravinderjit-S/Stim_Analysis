function [plv, f] = mtplv(x,params)
% Function to calculate across trial plv using multitaper method
%
% USAGE: [plv,f] = mtplv(x,params);
%
% x: input in channels x time x trials or time x trials
% params: A structure as in mtspectrumc() of the chronux toolbox i.e
%  params.Fs - Sampling rate
%  params.fpass - Frequency range of interest
%  params.pad - To zero pad or not to the next power of 2
%  params.tapers - [TW K]  i.e [time-half-BW product, #tapers]

if (ndims(x) == 3)
    nchans = size(x,1);
    ntime = size(x,2);
    ntrials = size(x,3);
else
    if(ndims(x) == 2)
        nchans = 1;
        ntime = size(x,1);
        ntrials = size(x,2);
        x = reshape(x,[1 ntime ntrials]);
    else
        fprintf(2,'\nSorry! The input data should be 2 or 3 dimensional!\n');
        help mtplv
        return;
    end
end

ntaps = params.tapers(2);
TW = params.tapers(1);
w = dpss(ntime,TW,ntaps);

if(params.pad == 1)
    nfft = 2^nextpow2(ntime);
else
    nfft = ntime;
end

plv = zeros(ntaps,nchans,nfft);

for k = 1:ntaps
    wbig = repmat(reshape(w(:,k),1,numel(w(:,k))),[nchans 1 ntrials]);
    X = fft(wbig.*x,nfft,2);
    plv(k,:,:) = squeeze(abs(mean(exp(1i*angle(X)),3)));
end
plv = mean(plv,1);
f = (0:(nfft-1))*params.Fs/nfft;

plv = plv(:,(f >= params.fpass(1))&(f <= params.fpass(2)));
f = f((f >= params.fpass(1))&(f <= params.fpass(2)));

    
    
    

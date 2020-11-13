function [stim,mods] = RandMod_Coherence(f1,f2,mod_cuts,bp_fo,fs,tlen,tinit,tincoh)

t = 0:1/fs:tlen-1/fs;

lb = mod_cuts(1);
ub = mod_cuts(2);
bp_filt = fir1(bp_fo, [lb ub]*2/fs, 'bandpass');

for i = 1:3

    noise1 = randn(1,1.5*length(t) + bp_fo + 1);
    noise2 = randn(1,1.5*length(t) + bp_fo + 1);
    noise3 = randn(1,1.5*length(t) + bp_fo + 1);

    noise_bp1 = filter(bp_filt,1,noise1);
    noise_bp2 = filter(bp_filt,1,noise2);
    noise_bp3 = filter(bp_filt,1,noise3);

    noise_bp1 = noise_bp1(bp_fo+1:bp_fo+length(t));
    noise_bp2 = noise_bp2(bp_fo+1:bp_fo+length(t));
    noise_bp3 = noise_bp3(bp_fo+1:bp_fo+length(t));

    noise_bp1 = noise_bp1 - min(noise_bp1);
    noise_bp1 = noise_bp1 / max(noise_bp1);
    noise_bp2 = noise_bp2 - min(noise_bp2);
    noise_bp2 = noise_bp2 / max(noise_bp2);
    noise_bp3 = noise_bp3 - min(noise_bp3);
    noise_bp3 = noise_bp3 / max(noise_bp3);
    

    tinit_samps = round(tinit*fs);
    tincoh_samps = round(tincoh*fs);

    if i==3
        mod1 = noise_bp1(1:tinit_samps);
        mod2 = noise_bp2(1:tinit_samps+tincoh_samps);
        mod1 = [mod1 noise_bp3(tinit_samps+1:end)];
        mod2 = [mod2 noise_bp3(tinit_samps+1:end-tincoh_samps)];
    else
        mod1 = noise_bp1(1:tinit_samps);
        mod2 = noise_bp2(1:tinit_samps);
        mod1 = [mod1 noise_bp3(tinit_samps+1:end)];
        mod2 = [mod2 noise_bp3(tinit_samps+1:end)];
    end
    
    lp_cut = lb+10;
    lp_filt = allPosFilt(lp_cut,fs);
    
    if length(lp_filt) > length(t)/2
        error('Low pass filter order too large')
    end
    
    mod1 = [mod1(round(length(t)/2):end), mod1, mod1(end-round(length(t)/2):end)];
    mod2 = [mod2(round(length(t)/2):end), mod2, mod2(end-round(length(t)/2):end)];
    mod1 = filtfilt(lp_filt,1,mod1);
    mod2 = filtfilt(lp_filt,1,mod2);
    
    mod1 = mod1(round(length(t)/2) +1:round(length(t)/2)  +length(t));
    mod2 = mod2(round(length(t)/2) +1:round(length(t)/2)  +length(t));
    

    min_mod = min([min(mod1),min(mod2)]);
    mod1 = mod1 - min_mod;
    mod2 = mod2 - min_mod;
    max_mod = max([max(mod1),max(mod2)]);
    mod1 = mod1 / max_mod;
    mod2 = mod2 / max_mod;


    
    x1 = sin(2*pi*f1.*t).*mod1;
    x2 = sin(2*pi*f2.*t).*mod2;
    stim{i} = x1+x2;
    mods{i} = [mod1;mod2];
end


end

function lp_filt = allPosFilt(lpf, fs)

    % Making a filter whose impulse response is purely positive (to avoid phase
    % jumps) so that the filtered envelope is purely positive. Using a dpss
    % window to minimize sidebands. For a bandwidth of bw, to get the shortest
    % filterlength, we need to restrict time-bandwidth product to a minimum.
    % Thus we need a length*bw = 2 => length = 2/bw (second). Hence filter
    % coefficients are calculated as follows:
    bw_lp = 2 * lpf;
    lp_filt = dpss(floor(2*fs/bw_lp),1,1);  % Using to increase actual bw when rounding
    lp_filt = lp_filt - lp_filt(1);
    lp_filt = lp_filt / sum(lp_filt);
end


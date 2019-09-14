function [envelope, lp_filt] = create_envelope(bw,lpf,Tlen,fs)
%This function creates an envelope by extracting an envelope from narrow
%band noise with bandwidth (bw). After nbn is generated, it is half-wave 
%rectified and low pass filtered to extract the envelope
%bw = 1x2 vector containing lower and upper bound of bandpass filter
%lpf = 1x1 bector containing upper bound of low pass
%Tlen = length of envelope in seconds
%fs = sampling rate in Hz

    T = 0:1/fs:Tlen-1/fs;
    noise = randn(1, 3*length(T)); %generating longer signal to chop off ends to deal with filter transients
    lb = bw(1); %lower bound
    ub = bw(2); %upper bound
    bp_fo = round(1/lb * 4 *fs); %filter order
    bp_filt = fir1(bp_fo, [lb*2/fs ub*2/fs], 'bandpass');

    noise_bp = fftfilt(bp_filt, noise);
    noise_bp = max(noise_bp, 0); %half-wave rectify bandpass noise

    lp_filt = allPosFilt(lpf, fs);
    envelope = fftfilt(lp_filt, noise_bp);
    envelope = envelope(length(T)+1:2*length(T));


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
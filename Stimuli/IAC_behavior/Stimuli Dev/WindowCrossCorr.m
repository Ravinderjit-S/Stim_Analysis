function [t, tlag, Wcc] = WindowCrossCorr(x, y, maxlag, wsize, wtype,overlap,fs, plotit)
%x & y are the 2 signals, wsize is the window size in samples, wtype is type of window
%overlap is amount of overlap in samples , fs = sampling rate to compute time
%t = time of signal, tlags = lag times (ITD for binaural signal), WCC =
%cross correlations at each t

%made for neural analysis of correlation and ITD in auditory system
%y is contralateral

if overlap > wsize
    error('Overlap can''t be greater than Wsize')
end
if length(x) ~= length(y) %account for this maybe ...
    error('x and y need to be same length right now')
end
if isempty(maxlag)
    maxlag = length(x);
end
if isempty(wtype)
    wtype = 'rect';
end
if size(x,1) > size(x,2)
    x = x';
end
if size(y,1) > size(y,2)
    y = y';
end

step = wsize - overlap;

t = (step:step:length(x)) /fs;

pad = zeros(1,wsize); %pad of zeros for last window(s) that may exceed length of signal
x = [x pad];
y = [y pad];

switch wtype
    case 'rect'
        windw = ones(1,wsize);
    case 'hann'
        windw = hann(wsize)';
    case 'hamm'
        windw = hamming(wsize)';
end
    
for i = 1:length(t)
    chunk = 1+(i-1)*step:(i-1)*step+wsize;
    sig1 = x(chunk).*windw;
    sig2 = y(chunk).*windw;
    [cc_i, lags_i] = xcorr(sig1, sig2,maxlag,'coeff');
    Wcc(:,i) = cc_i;
end

tlag = lags_i /fs;
tlag = tlag.*1e6;

if plotit
    figure()
    imagesc(t,tlag,Wcc, [-1,1]) ,colormap('jet')
    set(gca,'Ydir','normal')
    cbar = colorbar();
    cbar.Label.String = 'IAC';
    xlabel('time (s)')
    ylabel('ITD (us)')
    
    IAC_0ITD = Wcc(tlag ==0,:);
    figure, plot(t, IAC_0ITD), title('IAC at 0 ITD')
    xlabel('time (s)'), ylabel('IAC')
    [maxIAC, ITDloc] = max(Wcc);
    figure,plot(t,tlag(ITDloc)), title('ITD at max IAC'), xlabel('time (s)')
    ylabel('ITD (us)')
    figure,plot(t,maxIAC), title('max IAC'), xlabel('time (s)'), ylabel('IAC')
    
end

end


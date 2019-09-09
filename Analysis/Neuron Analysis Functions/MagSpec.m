function [f P1] = MagSpec(x,fs)
%this function returns the magnitude spectrum of signal x

fx = fft(x);
L = length(fx);
P1 = abs(fx(1:L/2+1))/L;
P1(2:end-1) = 2*P1(2:end-1);
f = fs*(0:L/2)/L;

end
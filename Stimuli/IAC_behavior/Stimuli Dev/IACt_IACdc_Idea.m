clear
fs = 48828.125;
t = 0:1/fs:1;
f = 50;
x = cos(2*pi*f*t);
window = ones(1,round(.500*fs));

[a,lags] = xcorr(x,window);

figure,plot(lags/fs,a) %ylim([-1 1])

% y = 1/exp(1) * exp(x); 
% figure,plot(t,y)
% b = xcorr(y,window);
% figure,plot(b)




% p = -1:.01:1; TnuTno = 1;
% y = -10*log10(p+(1-p)*TnuTno);
% y2 = p+(1-p)*TnuTno;
% figure,plot(p,-y2)
% 
% x1 = max(x,0);
% x2 = max(-x,0);
% y1 = x1+(1-x1)*TnuTno;
% y2 = x2+(1-x2)*TnuTno;
% y = y1-y2;
% 
% 
% figure,plot(t,y);
% 
% b = xcorr(y,window);
% figure,plot(b)
% 
% 

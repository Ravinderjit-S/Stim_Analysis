
A =9.108; 
f = 0:.1:20;
tau = 0.249; 

PSD = A.^2 ./ (1+4*(pi*f*tau).^2);

figure,plot(f,pow2db(PSD))


w = [0, 0.05,0.075,0.1,0.125,0.15,0.2,0.4,0.8,1.6];
IntWind = 0.1;
TnuTno =db2pow(13);
p = w./IntWind;
p(p>1) = 1;
p = -1:.001:1
IACfunc = -10*log10(p+(1-p)*TnuTno);

figure,plot(p,IACfunc)


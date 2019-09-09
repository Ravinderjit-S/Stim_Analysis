function y = BMLDFunc(w,TnuTno,T)

p = w/T;
p(p>1) = 1;
y = 10*log10(p+(1-p)*TnuTno);

end



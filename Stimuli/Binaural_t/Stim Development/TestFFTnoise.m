fs = 48828.125;
dur = 0.8;
flow = 500;
fhigh = 1500;
rng(5)
noise = makeNBNoiseFFT(1000,1000,dur,48828.125,[],0);
rng(5)
nbn = makeNBNfft_binaural_V2(flow,fhigh,dur,fs,1,0);

[p f] = pmtm(noise,2.5,[],fs);
[p2 f2] = pmtm(nbn(1,:),2.5,[],fs);

figure,plot(noise,'b')
figure,plot(nbn(1,:),'r'), title('nbn')
figure,plot(f,pow2db(p)), hold on, plot(f2,pow2db(p2),'r')








%Stimulus that looks at cortical activation to ITD changes and IPD changes

load('Seed16.mat'); %load s
rng(s);

ITD_s = 600e-6;
IPD_s = pi/2;

flow = 200;
fhigh = 1500;
dur1 = 1; 
dur2 = 0.5;
fs = 48828.125; 
rho = 1;

Total_dur = dur1+dur2;
t = 0:1/fs:Total_dur-1/fs;
AM = sin(2*pi*40*t-pi/2)/2+0.5;

Trials = 300;

for i =1:Trials
    nbn1 = makeNBNfft_binaural_V3(flow,fhigh,dur1,fs,rho,0,0);
    nbn2 = makeNBNfft_binaural_V3(flow,fhigh,dur2,fs,rho,ITD_s,0);
    stimITD = horzcat(nbn1,nbn2);
    stimITD = stimITD.*AM;
    save(['StimITD/stim_mid_lat_ITD' num2str(i)],'stimITD')
end
    

for i =1:Trials
    nbn1 = makeNBNfft_binaural_V3(flow,fhigh,dur1,fs,rho,0,0);
    nbn2 = makeNBNfft_binaural_V3(flow,fhigh,dur2,fs,rho,0,IPD_s);
    stimIPD = horzcat(nbn1,nbn2);
    stimIPD = stimIPD.*AM;
    save(['StimIPD/stim_mid_lat_IPD' num2str(i)],'stimIPD')
end


for i =1:Trials
    nbn1 = makeNBNfft_binaural_V3(flow,fhigh,dur1,fs,rho,-ITD_s,0);
    nbn2 = makeNBNfft_binaural_V3(flow,fhigh,dur2,fs,rho,ITD_s,0);
    stimITD2 = horzcat(nbn1,nbn2);
    stimITD2 = stimITD2.*AM;
    save(['StimITD2/stim_lat_lat_ITD2' num2str(i)],'stimITD2')
end

for i =1:Trials
    nbn1 = makeNBNfft_binaural_V3(flow,fhigh,dur1,fs,rho,0,-IPD_s);
    nbn2 = makeNBNfft_binaural_V3(flow,fhigh,dur2,fs,rho,0,IPD_s); 
    stimIPD2 = horzcat(nbn1,nbn2);
    stimIPD2 = stimIPD2.*AM;
    save(['StimIPD2/stim_lat_lat_IPD2' num2str(i)],'stimIPD2')
end




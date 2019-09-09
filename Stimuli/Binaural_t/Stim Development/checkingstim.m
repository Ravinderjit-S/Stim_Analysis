% checking stim
% clear
fs = 48828.125;
stimIAC=stimIAC_bs;
%[f p] = powerspec(stimIAC(1,:),fs);
%[f2 p2] = powerspec(-stimIAC(2,:),fs);
[p,f] = pmtm(stimIAC(1,:),2.5,[],fs);
[p2, f2] = pmtm(stimIAC(2,:),2.5,[],fs);

figure,plot(f,pow2db(p),'b',f2,pow2db(p2),'r')
figure,plot(stimIAC(1,:)), hold on, plot(stimIAC(2,:),'r')

%[f p] = powerspec(stimITD(1,:),fs);
%[f2 p2] = powerspec(stimITD(2,:),fs);
[p,f] = pmtm(stimITD(1,:),2.5,[],fs);
[p2,f2] = pmtm(stimITD(2,:),2.5,[],fs);

figure,plot(f,pow2db(p),'b',f2,pow2db(p2),'r')
figure,plot(stimITD(1,:)), hold on, plot(stimITD(2,:),'r')

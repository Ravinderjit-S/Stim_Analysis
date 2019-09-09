clear
load('Mseq_IAC_ITD.mat')

EEG_fs=4096;
Sound_fs = 48828.125;

IAC_val = 1;
ITD_val = 500e-6;
EachPoint = 2441/Sound_fs; %each point in M_seq is roughly 50ms
AddPoints_ideal = EachPoint*EEG_fs;
AddPoints = round(EachPoint*EEG_fs); 
CompensateEvery = ceil(1/(AddPoints - AddPoints_ideal));

Mseq_sig = [];
Mseq_sig2 = [];
for i =1:length(Mseq)
    if mod(i,CompensateEvery)==0
        Mseq_sig = [Mseq_sig Mseq(i) * ones(1,AddPoints-1)];
    else
        Mseq_sig = [Mseq_sig Mseq(i)*ones(1,AddPoints)];
    end
    Mseq_sig2 = [Mseq_sig2 Mseq(i)*ones(1,2441)];
end

t = 0:1/EEG_fs:(length(Mseq_sig)-1)/EEG_fs;
t2 = 0:1/Sound_fs:length(Mseq_sig2)/Sound_fs - 1/Sound_fs;

figure, plot(t,Mseq_sig,'b',t2,Mseq_sig2,'r')

mm = xcorr(Mseq_sig);

% [f,p]=MagSpec(Mseq_sig,EEG_fs); 
% figure,plot(f,p), title('MagSpec(Mseq)')

% [pxx,f] = pmtm(Mseq_sig,[],[],EEG_fs);
% figure, plot(f,pow2db(pxx)), title('pmtm(Mseq) db')

% [f,p]=MagSpec(mm,EEG_fs); 
% figure,plot(f,p)

% [pxx,f] = pmtm(mm,[],[],EEG_fs);
% figure, plot(f,pxx)






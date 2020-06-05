%Single-unit PLV
clear

addpath('../../Neuron Analysis Functions')
addpath('/media/ravinderjit/Data_Drive/Data/AuditoryNerve/DynBin')

Analyze_dates = [{'09.16.18'},{'09.20.18'},{'09.21.18'},{'10.01.18'},{'10.12.18'}];
date = Analyze_dates{3};
load(['MseqAnalyzed_' date '.mat'])

data = data_NOSCOR{1}; 

MSO = data.MSO.simulated;
stim_dur = data.Stimuli.Gating.Duration;
fs = 48828.125;
mseq = data.Stimuli.IACt;

H_imp = data.MSO.H_imp;
H_NF = data.MSO.H_NF;


for i = 1:length(MSO)
    plotRaster(MSO{i})
end

spks = MSO{end};
[t,MSO_fr] = FiringRate(spks,'bin', (1/fs)*1e3, stim_dur,0);
figure,plot(t,MSO_fr,t,(mseq+1)*400,'r')

figure, hold on
plot(H_imp{end})
plot(H_NF{end}{1},'r')

TW = 5;
ntaps = 2*TW-1;
w_res = TW/(length(mseq)/fs);
taps = dpss(length(mseq),TW,ntaps);


Trials = max(spks(:,1));
spk_trains = zeros(Trials,length(mseq));
for k = 1:Trials
    spk_times = spks(spks(:,1)==k,2);
    spk_times(spk_times > length(mseq)/fs) = [];
    spk_trains(k,round(spk_times*fs)) = 1;
end

nfft = 2^ceil(log2(length(mseq)));
plv_taps = zeros(ntaps,nfft);
mseq=(mseq+1)/2;
for m = 1:ntaps
    fprintf(1,'Taper %d \n',m)
    Xf = fft(spk_trains.*taps(:,m)',nfft,2);
    Mf = fft(mseq'.*taps(:,m)',nfft,2);
    XMf = Xf .* conj(Mf);
    plv_taps(m,:) = abs(mean(XMf ./ abs(XMf),1));
end
plv_taps = mean(plv_taps,1);
f = 0:fs/nfft:fs-1/nfft;

figure,plot(f,plv_taps)

H_imp = H_imp(length(mseq)+1:length(mseq)+0.1*fs);
pmtm(H_imp{end},2.5,[],fs)










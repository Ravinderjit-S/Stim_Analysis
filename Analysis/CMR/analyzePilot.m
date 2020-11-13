clear

Sig_4_p0 = load('CMR_SigAM_4phase_0.mat');
Sig_40_p0 = load('CMR_SigAM_40phase_0.mat');
Sig_223_p0 = load('CMR_SigAM_223phase_0.mat');

Sig_4_p180 = load('CMR_SigAM_4phase_180.mat');
Sig_40_p180 = load('CMR_SigAM_40phase_180.mat');
Sig_223_p180 = load('CMR_SigAM_223phase_180.mat');

figure, hold on
plot(Sig_4_p0.TrackSNRs,'r')
plot(Sig_40_p0.TrackSNRs,'b')
plot(Sig_223_p0.TrackSNRs,'k')
legend('4','40','223')
title('CoMod phase 0')

figure, hold on
plot(Sig_4_p180.TrackSNRs,'r')
plot(Sig_40_p180.TrackSNRs,'b')
plot(Sig_223_p180.TrackSNRs,'k')
legend('4','40','223')
title('Phase 180')

figure, hold on
plot(Sig_4_p0.TrackSNRs,'r')
plot(Sig_4_p180.TrackSNRs,'b')
title('Target Mod 4')

figure, hold on
plot(Sig_40_p0.TrackSNRs,'r')
plot(Sig_40_p180.TrackSNRs,'b')
title('Target Mod 40')

figure, hold on
plot(Sig_223_p0.TrackSNRs,'r')
plot(Sig_223_p180.TrackSNRs,'b')
title('Target Mod 223')

All_TrackSNRs = {Sig_4_p0.TrackSNRs,Sig_40_p0.TrackSNRs,Sig_223_p0.TrackSNRs, ...
    Sig_4_p180.TrackSNRs,Sig_40_p180.TrackSNRs,Sig_223_p180.TrackSNRs};

MedianSNR = [];
for j =1:length(All_TrackSNRs)
    TrackSNRs = All_TrackSNRs{j};
    changes = diff(TrackSNRs);
    changes_indexes = find(changes~=0);
    Reversal_spots = find(diff(sign(changes(changes~=0))));
    Reversal_SNRs = TrackSNRs(changes_indexes(Reversal_spots+1));
    Reversal_SNRs = [Reversal_SNRs TrackSNRs(end)]; %adding last reversal
    MedianSNR(j) = median(Reversal_SNRs(end-5:end));
end


CMR(1) = abs(MedianSNR(1) - MedianSNR(4));
CMR(2) = abs(MedianSNR(2) - MedianSNR(5));
CMR(3) = abs(MedianSNR(3) - MedianSNR(6));







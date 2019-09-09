subj = 'Rav';
load(['IAC_evoked20_' subj '.mat']) % this data is lowpassed at 50 Hz

fs = 4096;
splits = 15;
EEG_samps = size(IAC_EEG_avg,2);
t = 0:1/fs:EEG_samps/fs-1/fs;
% All_fits = [];
% for i =1:splits
%     Indexes = round(((i-1)/splits)*EEG_samps)+1:(i/splits)*EEG_samps;
%     Slope_fit = polyfit(t(Indexes),IAC_EEG_avg(31,Indexes),1);
%     Line_fit = t(Indexes)*Slope_fit(1)+Slope_fit(2);
%     All_fits = [All_fits,Line_fit];
% end

%IAC_EEG_avg = zscore(IAC_EEG_avg')';
% [Trend] = EEG_Trend(IAC_EEG_avg,fs,32,splits);
[Trend] = EEG_Trend_poly(IAC_EEG_avg,fs,32,8);
figure()
for i = 1:32
    subplot(8,4,i),plot(t,IAC_EEG_avg(i,:),t,Trend(i,:),'r',t,IAC_EEG_avg(i,:)-Trend(i,:),'k')
end


    
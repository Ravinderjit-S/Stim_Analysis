function [Trend] = EEG_Trend_poly(EEG_sig,fs,chans,order)

EEG_samps = size(EEG_sig,2);
t = 0:1/fs:EEG_samps/fs-1/fs;
for k =1:chans
    [FitVars] = polyfit(t,EEG_sig(k,:),order);
    Fit_poly = zeros(1,EEG_samps);
    for i = 1:length(FitVars)
        Fit_poly = Fit_poly+FitVars(i).* t.^(length(FitVars)-i);
    end
    Trend(k,:) = Fit_poly;
end
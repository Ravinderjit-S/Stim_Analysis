function [Trend] = EEG_Trend(EEG_sig,fs,chans,splits)
%splits is how many piecewise linear functions to split detrend into
%returns Trend, which is a piecewise function to remove trends in EEG
%One application is when one does not want to lpf and would rather DeTrend
%to remove DC offset and low frequency drifts
%EEG_sig is chans by samples
EEG_samps = size(EEG_sig,2);
t = 0:1/fs:EEG_samps/fs-1/fs;
for k = 1:chans
    All_fit = [];
    for i=1:splits
        Indexes = round(((i-1)/splits)*EEG_samps)+1:round((i/splits)*EEG_samps);
        Slope = polyfit(t(Indexes),EEG_sig(k,Indexes),1);
        Line_fit = t(Indexes)*Slope(1)+Slope(2);
        All_fit = [All_fit, Line_fit];
    end
    Trend(k,:) = All_fit;
end



end

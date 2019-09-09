function [BF, BF_RT, threshBF, Q10 ,unit, A_freqs,A_thresh] = plotTCs(data)
%Pass the cell array of pictures from NEL and this function will extract
%the tuninc curves and plot them
%BFs = best frequencies in KHz
%BF_RT = real time BFs (measured during exp)

TC_data = {};
for i = 1:numel(data)
    data_i = data{i};
    if isfield(data_i, 'TcData')
        TC_data = [TC_data data_i];
    end
end
figure, subplot(2,1,1), hold on
BF = [];
threshBF = [];
Q10 = nan(1,numel(TC_data));
unit = nan(1,numel(TC_data));

for j = 1:numel(TC_data)
    tc = TC_data{j};
    unit(j) = tc.General.track + tc.General.unit/100; 
    if tc.Thresh.thresh < 25
        warning(['Removed from analysis: bad Tuning curve in picture ' num2str(tc.General.picture_number)])
        continue
    end
    freqs_j = tc.TcData(:,1);
    thresh_j = tc.TcData(:,2);
    [~, BFrtInd] = min(-thresh_j);
    BF_RT(j) = freqs_j(BFrtInd);
    A_freqs{j} = freqs_j;
    A_thresh{j} = thresh_j;
    %ind_delete = find(thresh_j >20, 1, 'last');
    ind_delete = find(thresh_j(BFrtInd:end) <=20,1,'first')+BFrtInd-2;
    freqs_j = freqs_j(1:ind_delete); thresh_j = thresh_j(1:ind_delete); %chop off part after tuning curve is obtained
    ind_delete = find(thresh_j <=20,1,'last') + 1;
    freqs_j = freqs_j(ind_delete:end); thresh_j = thresh_j(ind_delete:end); % chop off beginning part before tuning curve
    
    
    frequencies = freqs_j(end):.01:freqs_j(1);
    tc_interp = interp1(freqs_j, thresh_j, frequencies,'spline');
    [thresh_BF BF_ind] = max(tc_interp);
    BF = [BF frequencies(BF_ind)];
    threshBF = [threshBF thresh_BF]; 
   
    
    Q10bw_ind1 = find(tc_interp(1:BF_ind) <= threshBF(end)-10,1,'last')+1;

    Q10bw_ind2 = find(tc_interp(Q10bw_ind1:end) <= threshBF(end)-10,1) + Q10bw_ind1;
  
    Q10bw = frequencies(Q10bw_ind2) - frequencies(Q10bw_ind1);
    Q10(j) = BF(end) / Q10bw;
    semilogx(frequencies*1000,-tc_interp,'Color', [rand rand rand]) % plot tuning curve
%     plot(tc.TcData(:,1),-tc.TcData(:,2),'rx')
%     plot(freqs_j, -thresh_j, 'bx')
    
end
semilogx(BF*1000,-threshBF,'or')
xlim([0 max(BF)*1000+2000])
xlabel('KHz'), ylabel('dB'), title('Tuning Curves')
hold off

subplot(2,1,2), semilogx(BF*1e3, Q10,'rx'), xlabel('BF (Hz)'), ylabel('Q10')


end
    
    
        

clear all
dur = 1;
fs = 48828.125;
%BPfilt = designfilt('bandpassiir', 'StopbandFrequency1', 190, 'PassbandFrequency1', 200, 'PassbandFrequency2', 1500, 'StopbandFrequency2', 1550, 'StopbandAttenuation1', 30, 'PassbandRipple', 1, 'StopbandAttenuation2', 30, 'SampleRate', fs);
BPfilt = designfilt('bandpassfir', 'StopbandFrequency1', 100, 'PassbandFrequency1', 200, 'PassbandFrequency2', 1500, 'StopbandFrequency2', 2000, 'StopbandAttenuation1', 60, 'PassbandRipple', 1, 'StopbandAttenuation2', 60, 'SampleRate', fs);

fm = 30;
tic()
stim = IACsinAFC3(dur,fs,fm,BPfilt);
toc()
a = randperm(3);
for i =1:3
    play = stim{a(i)};
    play(1,:) = rampsound(play(1,:),fs,.150); 
    play(2,:) = rampsound(play(2,:),fs,.150);
    play(1,:) = play(1,:) ./ rms(play(1,:)); play(2,:) = play(2,:) ./ rms(play(2,:));
    sound(play/8,fs)
    pause(2)
end

% maxlag = round(6.5e-4*fs); wsize = ceil(.002*fs); wtype = 'hann';
% plotit =1; overlap = round(wsize/2);
% [tt, tlag, Wcc] = WindowCrossCorr(stim{3}(1,:),stim{3}(2,:),maxlag,wsize,wtype,overlap,fs,plotit);
% sound(stim{3},fs)

% 
% Grpdly = grpdelay(BPfilt);Grpdly = round(Grpdly(1));
% StimOscor = stim{2};
% t = 0:1/fs:length(stim{1}(1,:))/fs-1/fs;
% A = cos(2*pi*fm.*t);
% t = t*1000;
% fig =figure; 
% left_color = [0 0 0];
% right_color = [0.4 0.4 0.4];
% set(fig,'defaultAxesColorOrder',[left_color; right_color]);
% 
% FlatIAC = zeros(1,length(A));
% 
% yyaxis left, ylim([-1 1]), ylabel('IAC'), yticks([-1 0 1])
% plot(t(Grpdly:end)-t(Grpdly),FlatIAC(1:end-Grpdly+1),'k','linewidth',2)
% yyaxis right, ylim([-.9 .9]), ylabel('Signal Amplitude')
% hold on
% plot(t(Grpdly:end)-t(Grpdly),StimOscor(1,Grpdly:end),'b','linewidth',2)
% plot(t(Grpdly:end)-t(Grpdly),StimOscor(2,Grpdly:end),'r-','linewidth',2)
% hold off
% xlim([0 1/fm]*1000)
% legend('IAC','Left Stim', 'Right Stim','location','Northeast')
% xlabel('Time (ms)')
% set(gca,'fontsize',25)
% fig = gcf;
% fig.PaperUnits = 'inches';
% fig.PaperPosition = [0 0 11 8];
% print('OscorFMStim_unCorr2','-dpng','-r0')







clear

subj = 'Rav';
load(['IAC_evoked20_' subj '.mat'])
load(['ITD_evoked20_' subj '.mat'])

load('Mseq_4096fs.mat') %Mseq_sig
%load('Mseq_dumb.mat') %Mseq_dumb
Aud_channels = [5,31,32,26,22,13,9];
fs =4096;
Mseq_ITD = Mseq_sig;



% for jj = 1:size(Mseq_dumb,1) %10 and 10r + 1r dummies
%     for ii=1:32 %generate dummies for every channel
%         [Rev_IAC_di, ~] = xcorr(IAC_EEG_avg(ii,:), Mseq_dumb(jj,1:end), 2*round(fs),'coeff');
%         [Rev_ITD_di, ~] = xcorr(ITD_EEG_avg(ii,:), Mseq_dumb(jj,1:end), 2*round(fs),'coeff');
%         Rev_IAC_dummies(jj,:,ii) = Rev_IAC_di;
%         Rev_ITD_dummies(jj,:,ii) = Rev_ITD_di;
%     end
% end

for i = 1:32
    [Rev_IAC_i, lags] = xcorr(IAC_EEG_avg(i,:),Mseq_sig,2*round(fs),'coeff');
    [Rev_ITD_i, ~] = xcorr(ITD_EEG_avg(i,:),Mseq_ITD,2*round(fs),'coeff');
    t = lags/fs;
    Rev_IAC(i,:) = Rev_IAC_i;
    Rev_ITD(i,:) = Rev_ITD_i;
end
%save([subj '_RevCorMseq.mat'],'Rev_IAC_dummies','Rev_ITD_dummies','Rev_IAC','Rev_ITD')
    
figure()
for pp = 1:32
    subplot(8,4,pp)
%     hold on, plot(ones(1,21)*.20,[-0.1:.01:0.1],'k'), plot(zeros(1,21),[-0.1:.01:0.1],'k'),hold off
%     hold on
%     for kk = 1:size(Rev_IAC_dummies,1)
%         plot(t,Rev_IAC_dummies(kk,:,pp),'Color',[0 1 1])
%     end
    plot(t,Rev_IAC(pp,:),'b'),title(num2str(pp)),ylim([-0.15,0.15]),xlim([0 1])
%     hold off
    if any(pp==Aud_channels)
        set(gca,'Color','y')
    end
end
figure()
for pp = 1:32
    subplot(8,4,pp)
%     hold on, plot(ones(1,21)*.25,[-0.1:.01:0.1],'r'), hold off
%     hold on
%     for kk=1:size(Rev_ITD_dummies,1)
%         plot(t,Rev_ITD_dummies(kk,:,pp),'Color',[0 1 1])
%     end
    plot(t,Rev_ITD(pp,:),'b'),title(num2str(pp)), ylim([-0.07,0.07])
%     hold off
    if any(pp==Aud_channels)
        set(gca,'Color','y')
    end
end

figure()
hold on
% [pIAC, f] = pmtm(Rev_IAC(32,:),5,0:1:50,fs);
for pp=1:32
    subplot(8,4,pp)
%     hold on
%     for kk=1:size(Rev_IAC_dummies,1)
%         [pdumb, ~] = pmtm(Rev_IAC_dummies(kk,:,32),5,0:1:50,fs);
%         plot(f,pow2db(pdumb),'Color',[0 1 1])
%     end
    [pIAC, f] = pmtm(Rev_IAC(pp,:),2.5,[],fs);
    plot(f,pow2db(pIAC),'b'),title(num2str(pp)), xlim([0 30])
%     hold off
    if any(pp==Aud_channels)
        set(gca,'Color','y')
    end
end
    



% [pxx,f] =pmtm(r,2,[],4096);
% [pjj,f2] = pmtm(j,2,[],4096);
% figure,plot(f,pow2db(pxx),'b',f2,pow2db(pjj),'r'),title('pmtm(revcor)')
% 
% 
% [aa lags] = xcorr(Mseq_sig, Mseq_sig(end:-1:1));
% figure,plot(lags,aa)
% 
[pzz, f3] = pmtm(IAC_EEG_avg(32,:),2.5,[],4096);
figure,plot(f3,pow2db(pzz)),title('EEG spect')

 




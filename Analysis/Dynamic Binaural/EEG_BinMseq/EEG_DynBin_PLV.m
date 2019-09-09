clear

%load('Mseq_4096fs.mat') %Mseq_sig
%load('Mseq_dumb.mat') %Mseq_dumb
load('Mseq_4096fs_compensated.mat')

Subjects = [{'Rav'},{'S203'},{'S204'},{'S132'},{'S206'},{'S207'}];
subj = Subjects{4};
% load(['IAC_evoked20_' subj '.mat']) 
% load(['ITD_evoked20_' subj '.mat'])
load(['IAC_epochs20_' subj '.mat'])
load(['ITD_epochs20_' subj '.mat']) 

fs = 4096;
Mseq_ITD = Mseq_sig; 

IAC_EEG_epochs = IAC_EEG_epochs(:,:,1:length(Mseq_sig));
% IAC_EEG_avg = IAC_EEG_avg(:,1:length(Mseq_sig));
ITD_EEG_epochs = ITD_EEG_epochs(:,:,1:length(Mseq_sig));
% ITD_EEG_avg = ITD_EEG_avg(:,1:length(Mseq_sig));

IACeeg_ch = reshape(IAC_EEG_epochs(:,32,:),[size(IAC_EEG_epochs,1),size(IAC_EEG_epochs,3)]);
ITDeeg_ch = reshape(ITD_EEG_epochs(:,32,:),[size(ITD_EEG_epochs,1),size(ITD_EEG_epochs,3)]);
for i = 1:size(IACeeg_ch,1)
    [Rev_IAC_i] = cconv(IACeeg_ch(i,:),Mseq_sig(end:-1:1));
    Rev_IAC(i,:) = Rev_IAC_i(length(Mseq_sig):end);

    if mod(i,50) == 0
        fprintf('Done with %d\n',i)
    end
end
Rev_IAC = Rev_IAC';

for i =1:size(ITDeeg_ch,1)
    [Rev_ITD_i] = cconv(ITDeeg_ch(i,:),Mseq_ITD(end:-1:1));
    Rev_ITD(i,:) = Rev_ITD_i(length(Mseq_sig):end);
    if mod(i,50) == 0
        fprintf('Done with %d\n',i)
    end
end
Rev_ITD = Rev_ITD';

Keep_t = 1; %amount of time of impulse response to keep
Rev_IAC = Rev_IAC(1:round(Keep_t*fs),:);
params.Fs = fs;
params.fpass = [0, 40];
params.pad = 1;
params.tapers = [5, 10];
[plv, f] = mtplv(Rev_IAC,  params);

figure,plot(f,plv)

Rev_ITD = Rev_ITD(1:round(Keep_t*fs),:);
[plv,f] = mtplv(Rev_ITD,params);

figure,plot(f,plv)



t=0:1/fs:52224/fs-1/fs;
ch32 = reshape(IAC_EEG_epochs(:,32,:),[size(IAC_EEG_epochs,3),size(IAC_EEG_epochs,1)]);
figure,plot(t,ch32(:,4))






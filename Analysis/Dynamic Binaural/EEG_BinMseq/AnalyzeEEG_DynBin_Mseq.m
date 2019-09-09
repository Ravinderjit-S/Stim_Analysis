function [] = AnalyzeEEG_DynBin_Mseq(subj,Aud_channels,Num_noiseFloors,Keep_H)
%subj = subject ID
%Aud_channels = channels to analyze
%Num_noiseFloors = # of noise floors
%Keep_H = amount of time of impulse response to keep

ITD_pow20 = [];
IAC_pow20 = [];

load('Mseq_4096fs_compensated.mat')
Mseq_ITD = Mseq_sig; 
fs = 4096; % EEG fs

load(['IAC_evoked20_' subj '.mat']) 
load(['ITD_evoked20_' subj '.mat'])
load(['IAC_epochs20_' subj '.mat'])
load(['ITD_epochs20_' subj '.mat'])

IAC_EEG_epochs = IAC_EEG_epochs(:,:,1:length(Mseq_sig));
IAC_EEG_avg = IAC_EEG_avg(:,1:length(Mseq_sig));
ITD_EEG_epochs = ITD_EEG_epochs(:,:,1:length(Mseq_sig));
ITD_EEG_avg = ITD_EEG_avg(:,1:length(Mseq_sig));

%% Visualize Data

figure(), hold on
t = 0:1/fs:(size(ITD_EEG_avg,2)-1)/fs;
for i=1:32
%     IAC_EEG_avg(i,:) = filtfilt(HPfilter,IAC_EEG_avg(i,:));
%     Trend = EEG_Trend(IAC_EEG_avg,fs,32,order);
%     IAC_EEG_avg = IAC_EEG_avg(1:32,:) - Trend;
    subplot(8,4,i),plot(t,IAC_EEG_avg(i,:)),title(['IAC' num2str(i)]),xlim([0 2])
    if any(i==Aud_channels)
        set(gca,'Color','y')
    end
end
hold off


figure(), hold on
t = 0:1/fs:(size(ITD_EEG_avg,2)-1)/fs;
for i=1:32
%     IAC_EEG_avg(i,:) = filtfilt(HPfilter,IAC_EEG_avg(i,:));
%     Trend = EEG_Trend(IAC_EEG_avg,fs,32,order);
%     IAC_EEG_avg = IAC_EEG_avg(1:32,:) - Trend;
    [p, f] = pmtm(IAC_EEG_avg(i,:),2.5,[],fs);
%     IAC_pow20(i) = abs(pow2db(p(find(f>=20,1)))) - abs(mean(pow2db(p(f>=16&f<=19))));
    subplot(8,4,i),plot(f,pow2db(p)),title(['IAC' num2str(i)]),xlim([0 50])
    if any(i==Aud_channels)
        set(gca,'Color','y')
    end
end
% [IAC_pow20Sorted, SortInds] = sort(IAC_pow20);
% Aud_channels = SortInds(1:6);

figure(), hold on
for i=1:32
%      ITD_EEG_avg(i,:) = filtfilt(HPfilter,ITD_EEG_avg(i,:));
%     Trend = EEG_Trend(ITD_EEG_avg,fs,32,order);
%     ITD_EEG_avg = ITD_EEG_avg(1:32,:) - Trend;
    subplot(8,4,i),plot(t,ITD_EEG_avg(i,:)),title(['ITD' num2str(i)]),xlim([0 2])
    if any(i==Aud_channels)
        set(gca,'Color','y')
    end
end
hold off

figure(), hold on
for i=1:32
%     IAC_EEG_avg(i,:) = filtfilt(HPfilter,IAC_EEG_avg(i,:));
%     Trend = EEG_Trend(IAC_EEG_avg,fs,32,order);
%     IAC_EEG_avg = IAC_EEG_avg(1:32,:) - Trend;
    [p, f] = pmtm(ITD_EEG_avg(i,:),2.5,[],fs);
%     ITD_pow20(i) = abs(pow2db(p(find(f>=20,1)))) - abs(mean(pow2db(p(f>=16&f<=19))));
    subplot(8,4,i),plot(f,pow2db(p)),title(['ITD' num2str(i)]),xlim([0 50])
    if any(i==Aud_channels)
        set(gca,'Color','y')
    end
end
hold off

fprintf('Done loading and visualizing data \n')

%% Analyze only Auditory Channels

IAC_EEG_avg = IAC_EEG_avg(Aud_channels,:);
ITD_EEG_avg = ITD_EEG_avg(Aud_channels,:);
IAC_EEG_epochs = IAC_EEG_epochs(:,Aud_channels,:);
ITD_EEG_epochs = ITD_EEG_epochs(:,Aud_channels,:);

%% Generate Noise Floors 

for jj = 1:Num_noiseFloors
    epochs_num_IAC = size(IAC_EEG_epochs,1);
    RandInds_IAC = randperm(epochs_num_IAC);
    epochs_num_ITD = size(ITD_EEG_epochs,1);
    RandInds_ITD = randperm(epochs_num_ITD);
    
    Noise_EEG_IAC = (sum(IAC_EEG_epochs(RandInds_IAC(1:round(epochs_num_IAC/2)),:,:)) - sum(IAC_EEG_epochs(RandInds_IAC(round(epochs_num_IAC/2)+1:end),:,:)))./epochs_num_IAC; %adding half of one with the other
    Noise_EEG_IAC = reshape(Noise_EEG_IAC,[numel(Aud_channels) length(Mseq_sig)]);
    Noise_EEG_ITD = (sum(ITD_EEG_epochs(RandInds_ITD(1:round(epochs_num_ITD/2)),:,:)) - sum(ITD_EEG_epochs(RandInds_ITD(round(epochs_num_ITD/2)+1:end),:,:)))./epochs_num_ITD;
    Noise_EEG_ITD = reshape(Noise_EEG_ITD,[numel(Aud_channels) length(Mseq_sig)]);
    
    for kk =1:numel(Aud_channels)
        [Rev_IAC_dumbkk] = cconv(Noise_EEG_IAC(kk,:),Mseq_sig(end:-1:1));
        [Rev_ITD_dumbkk] = cconv(Noise_EEG_ITD(kk,:),Mseq_sig(end:-1:1));
        Rev_IACdumb(kk,:) = Rev_IAC_dumbkk(length(Mseq_sig):end); %#ok
        Rev_ITDdumb(kk,:) = Rev_ITD_dumbkk(length(Mseq_sig):end); %#ok
    end
    
    NoiseFloors_IAC{jj} = Rev_IACdumb; %#ok
    NoiseFloors_ITD{jj} = Rev_ITDdumb; %#ok
    
end

fprintf('Done Generating Noise Floors \n')

%% Generate Transfer function estimates

for i = 1:numel(Aud_channels)
    [Rev_IAC_i] = cconv(IAC_EEG_avg(i,:),Mseq_sig(end:-1:1));
    [Rev_ITD_i] = cconv(ITD_EEG_avg(i,:),Mseq_ITD(end:-1:1));
    Rev_IAC(i,:) = Rev_IAC_i(length(Mseq_sig):end); % extract valid part
    Rev_ITD(i,:) = Rev_ITD_i(length(Mseq_sig):end); 
end

fprintf('Done Generating H estimates \n')

%% Plot H(t)

t = (0:length(Mseq_sig)-1)/fs;

figure()
if isprime(numel(Aud_channels))
    plottxy=factor(numel(Aud_channels)+1);
else
    plottxy = factor(numel(Aud_channels));
end
plottingX = prod(plottxy(1:round(numel(plottxy)./2)));
plottingY = prod(plottxy(round(numel(plottxy)./2)+1:end));
for pp = 1:numel(Aud_channels)
    subplot(plottingX,plottingY,pp)
    hold on
    for kk = 1:Num_noiseFloors
        dummy = NoiseFloors_IAC{kk};
        plot(t,dummy(pp,:),'Color',[0 1 1])
    end
    plot(t,Rev_IAC(pp,:),'b'),title(['IAC H(t):' num2str(Aud_channels(pp))]), xlim([0 Keep_H])
    hold off
end

figure()
for pp = 1:numel(Aud_channels)
    subplot(plottingX,plottingY,pp)
    hold on
    for kk=1:Num_noiseFloors
        dummy = NoiseFloors_ITD{kk};
        plot(t,dummy(pp,:),'Color',[0 1 1])
    end
    plot(t,Rev_ITD(pp,:),'b'),title(['ITD H(t):' num2str(Aud_channels(pp))]), xlim([0 Keep_H])
    hold off
end

%% Plot H(f)

figure()
for pp= 1:numel(Aud_channels)
    subplot(plottingX,plottingY,pp)
    hold on
    for kk=1:Num_noiseFloors
        dummy = NoiseFloors_IAC{kk};
        [pdumb, f] = pmtm(dummy(pp,1:round(Keep_H*fs)),2.5,[],fs);
        plot(f,pow2db(pdumb),'Color',[0 1 1])
    end
    [pIAC, f] = pmtm(Rev_IAC(pp,1:round(Keep_H*fs)),2.5,[],fs);
    plot(f,pow2db(pIAC),'b'),title(['IAC H(f)' num2str(Aud_channels(pp))]), xlim([0 20])
    hold off
end

   
figure()
for pp= 1:numel(Aud_channels)
    subplot(plottingX,plottingY,pp)
    hold on
    for kk=1:Num_noiseFloors
        dummy = NoiseFloors_ITD{kk};
        [pdumb, f] = pmtm(dummy(pp,1:round(Keep_H*fs)),2.5,[],fs);
        plot(f,pow2db(pdumb),'Color',[0 1 1])
    end
    [pITD,f] = pmtm(Rev_ITD(pp,1:round(Keep_H*fs)),2.5,[],fs);
    plot(f,pow2db(pITD),'b'),title(['ITD H(f)' num2str(Aud_channels(pp))]),xlim([0 20])
    hold off
end

%% Save variables
save([subj '_DynBinMseqAnalyzed.mat'],'Rev_IAC','Rev_ITD','NoiseFloors_IAC','NoiseFloors_ITD','Aud_channels')





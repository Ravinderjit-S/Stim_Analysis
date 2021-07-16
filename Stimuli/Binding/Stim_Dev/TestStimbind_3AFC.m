fs = 48828;
%Corr_inds = [12:16];
Corr_inds = randperm(20);
Corr_inds = Corr_inds(11:20);

bw = [4, 24];

tic()
[stim_Ref, stimA, stimB1, stimA2, ERBspace, Tones_f, envsA, envsB] = StimBind_noRect_3AFC(Corr_inds, fs,bw);
toc()

t = 0:1/fs:length(stimB1)/fs - 1/fs;
figure,
hold on
plot(t,stim_Ref)
plot(t,stimB1+10)

figure,
plot(t, envsA')

figure,
plot(t,envsB')

figure,
pmtm(envsA(1,:) - mean(envsA(1,:)), 2.5, 1:100,fs)

figure()
spectrogram(stim_Ref,fs/20,[],[],fs,'yaxis'), ylim([0 10])
caxis([-45,-40])

figure()
spectrogram(stimB1,fs/20,[],[],fs,'yaxis'), ylim([0 10])
caxis([-45,-40])

figure()
pmtm(stimB1, 2.5,100:9000,fs)

% figure()
% spectrogram(stimA,fs/100,[],[],fs,'yaxis'), ylim([0 10])


soundsc(stim_Ref,fs)
pause(1.5)
soundsc(stimB1,fs)
pause(1.5)
soundsc(stimA,fs)


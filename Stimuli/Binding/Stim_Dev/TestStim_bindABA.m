fs = 48828;
f_start = 200;
f_end = 8000; 
Tones_num = 20;
%Corr_inds = [15:20];
%Corr_inds = [2,8,14,20];
%Corr_inds = [1,4,7,10,13,16];
Corr_inds = [1,4,8,12,16,20];
ERB_spacing = []; %if specified, takes precedence over Tones_num

tic()
[stim_Ref, stimA, stimB, stimA2, envs_A, envs_B, ~, ERBspace, Tones_f] = Stim_Bind_ABA(Corr_inds, fs, f_start, f_end, Tones_num, ERB_spacing);
toc()

stims = vertcat(stim_Ref,stimA,stimA2, stimB);
order = randperm(3);
order = [1 order+1];

for i = 1:3
    soundsc(stims(order(i),:),fs)
    pause(1.5)
end

t = 0:1/fs:1.0-1/fs;
figure
ax1 = subplot(3,1,1);
plot(t,stimA), title('A')

ax2 = subplot(3,1,2);
plot(t,stimB), title('B1')

ax3 = subplot(3,1,3);
plot(t,stimA2), title('A2')

linkaxes([ax1,ax2,ax3],'xy')
%ylim([-0.8,0.8])

figure
spectrogram(stimB,round(fs*.01),round(fs*.01*.9),[1:9000],fs,'yaxis')


figure,plot(t,envs_B(1,:),'b')

% figure,plot(t,stimB)






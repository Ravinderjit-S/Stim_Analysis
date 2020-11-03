fs = 48828;
Corr_inds = 1:6 ;
f_start = 100;
f_end = 8000; 
Tones_num = 16;
ERB_spacing = []; %if specified, takes precedence over Tones_num
tic()
[stimA, stimB, stimA2, ~, ~, ~, ERBspace, Tones_f] = Stim_Bind_ABA(Corr_inds, fs, f_start, f_end, Tones_num, ERB_spacing);
toc()
stims = vertcat(stimA,stimA2, stimB);
order = randperm(3);

for i = 1:3
    sound(stims(order(i),:),fs)
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
ylim([-0.4,0.4])

figure
spectrogram(stimB,fs/50,[],[],fs,'yaxis'), ylim([0 10])


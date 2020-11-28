fs = 48828;
f_start = 600;
f_end = 8000; 
Tones_num = 8;
Corr_inds = [1:Tones_num];
bw = [103 111]; % 4, 24
bp_fo = 1/2 * 4 * fs; %round(1/lb * 4 *fs)
lpf = []; %40
ERB_spacing = []; %if specified, takes precedence over Tones_num
tic()
[stim_Ref, stimA, stimB, stimA2, envs_A, envs_B, ~, ERBspace, Tones_f] = Stim_Bind_ABA(Corr_inds, fs, f_start, f_end, Tones_num, ERB_spacing,bw,bp_fo,lpf);
toc()
stims = vertcat(stim_Ref,stimA,stimA2, stimB);
order = randperm(3);
order = [1 order+1];

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
%ylim([-0.8,0.8])

figure
spectrogram(stimB,round(fs*.01),round(fs*.01*.9),[1:9000],fs,'yaxis')


figure,plot(t,envs_B(1,:),'b')


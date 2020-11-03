fs = 48828;
f_start = 400;
f_end = 6000; 
Tones_num = 3;
ERB_spacing = 2; %if specified, takes precedence over Tones_num

[Tones_f, ERBspace] = Get_Tones(Tones_num, ERB_spacing, f_start, f_end); 

% Tones_f_Int = Tones_f(2:6);
% Corr_inds_Int =  [1,3,5];
% 
% Tones_f_noInt = Tones_f([1,2,4,6,7]);
% Corr_inds_noInt = [2:4];

Tones_f_Int = Tones_f(3:9);
Corr_inds_Int = [1,4,7];

Tones_f_noInt = Tones_f([1,2,3,6,9,10,11]);
Corr_inds_noInt = [3:5];



tic()
[stimA, stimB, stimA2, ~, ~, ~] = Stim_Bind_ABA_giveFreq(Corr_inds_Int, fs, Tones_f_Int);
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
spectrogram(stimB,fs/50,[],[],fs,'yaxis'), ylim([0 6])


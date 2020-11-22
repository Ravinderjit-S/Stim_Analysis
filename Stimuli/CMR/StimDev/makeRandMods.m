% making a bunch of random modulations because this takes forever
clear
fs = 44100;
bp_mod_fo = 1/2 * 5 *fs;
n_mod_cuts = [2 10]
tlen = 1;
t = 0:1/fs:tlen-1/fs;
bp_filt_mod = fir1(bp_mod_fo, n_mod_cuts * 2/fs,'bandpass');


% for i = 1:300
%     noise_mod_i = randn(1, 1.5*length(t)+bp_mod_fo+1);
%     noise_mod_i = filter(bp_filt_mod,1,noise_mod_i);
%     
%     noise_mod_i = noise_mod_i(bp_mod_fo+1:bp_mod_fo+length(t));
%     noise_mod_i = noise_mod_i - min(noise_mod_i);
%     noise_mod(i,:) = noise_mod_i / max(noise_mod_i);
%     fprintf(1,'%d/300 \n',i)
% end
% 
% save(['RandMod_' num2str(n_mod_cuts(2)) '.mat'],'noise_mod','n_mod_cuts','fs');

n_mod_cuts = [32 40]
noise_mod = [];
bp_filt_mod = fir1(bp_mod_fo, n_mod_cuts * 2/fs,'bandpass');
for i = 1:300
    noise_mod_i = randn(1, 1.5*length(t)+bp_mod_fo+1);
    noise_mod_i = filter(bp_filt_mod,1,noise_mod_i);
    
    noise_mod_i = noise_mod_i(bp_mod_fo+1:bp_mod_fo+length(t));
    noise_mod_i = noise_mod_i - min(noise_mod_i);
    noise_mod(i,:) = noise_mod_i / max(noise_mod_i);
    fprintf(1,'%d/300 \n',i)
end

save(['RandMod_' num2str(n_mod_cuts(2)) '.mat'],'noise_mod','n_mod_cuts','fs');


n_mod_cuts = [125 132]
noise_mod = []
bp_filt_mod = fir1(bp_mod_fo, n_mod_cuts * 2/fs,'bandpass');
for i = 1:300
    noise_mod_i = randn(1, 1.5*length(t)+bp_mod_fo+1);
    noise_mod_i = filter(bp_filt_mod,1,noise_mod_i);
    
    noise_mod_i = noise_mod_i(bp_mod_fo+1:bp_mod_fo+length(t));
    noise_mod_i = noise_mod_i - min(noise_mod_i);
    noise_mod(i,:) = noise_mod_i / max(noise_mod_i);
    fprintf(1,'%d/300 \n',i)
end

save(['RandMod_' num2str(n_mod_cuts(2)) '.mat'],'noise_mod','n_mod_cuts','fs');











% making a bunch of random modulations because this takes forever
clear
fs = 44100;
bp_mod_fo = 1/2 * 5 *fs;
n_mod_cuts = [2 10]
tlen = 1;
t = 0:1/fs:tlen-1/fs;
bp_filt_mod = fir1(bp_mod_fo, n_mod_cuts * 2/fs,'bandpass');

lp_co = n_mod_cuts(2) + log10(n_mod_cuts(2))*20;
bw_lp = 2*lp_co;
lp_filt = dpss(floor(2*fs/bw_lp),1,1);  % Using to increase actual bw when rounding
lp_filt = lp_filt - lp_filt(1);
lp_filt = lp_filt / sum(lp_filt);

noise_mod = zeros(200,length(t));

for i = 1:200
    noise_mod_i = randn(1, 1.5*length(t)+bp_mod_fo+1+length(lp_filt));
    noise_mod_i = filter(bp_filt_mod,1,noise_mod_i);
    noise_mod_i = noise_mod_i(bp_mod_fo+1:end);
    
    noise_mod_i = max(noise_mod_i,0); %h rectify
    noise_mod_i = filter(lp_filt,1,noise_mod_i);
    noise_mod_i = noise_mod_i(length(lp_filt)+1:length(lp_filt)+length(t));
    
    noise_mod(i,:) = noise_mod_i / max(noise_mod_i);
    fprintf(1,'%d/200 \n',i)
end

save(['RandMod_Hrect_' num2str(n_mod_cuts(2)) '.mat'],'noise_mod','n_mod_cuts','fs');

n_mod_cuts = [32 40]
noise_mod = zeros(200,length(t));
bp_filt_mod = fir1(bp_mod_fo, n_mod_cuts * 2/fs,'bandpass');

lp_co = n_mod_cuts(2) + log10(n_mod_cuts(2))*20;
bw_lp = 2*lp_co;
lp_filt = dpss(floor(2*fs/bw_lp),1,1);  % Using to increase actual bw when rounding
lp_filt = lp_filt - lp_filt(1);
lp_filt = lp_filt / sum(lp_filt);


for i = 1:200
    noise_mod_i = randn(1, 1.5*length(t)+bp_mod_fo+1);
    noise_mod_i = filter(bp_filt_mod,1,noise_mod_i);
    noise_mod_i = noise_mod_i(bp_mod_fo+1:end);
    
    noise_mod_i = max(noise_mod_i,0); %h rectify
    noise_mod_i = filter(lp_filt,1,noise_mod_i);
    noise_mod_i = noise_mod_i(length(lp_filt)+1:length(lp_filt)+length(t));
    
    noise_mod(i,:) = noise_mod_i / max(noise_mod_i);
    fprintf(1,'%d/200 \n',i)
end

save(['RandMod_Hrect_' num2str(n_mod_cuts(2)) '.mat'],'noise_mod','n_mod_cuts','fs');


n_mod_cuts = [125 132]
noise_mod = zeros(200,length(t));
bp_filt_mod = fir1(bp_mod_fo, n_mod_cuts * 2/fs,'bandpass');

lp_co = n_mod_cuts(2) + log10(n_mod_cuts(2))*20;
bw_lp = 2*lp_co;
lp_filt = dpss(floor(2*fs/bw_lp),1,1);  % Using to increase actual bw when rounding
lp_filt = lp_filt - lp_filt(1);
lp_filt = lp_filt / sum(lp_filt);


for i = 1:200
    noise_mod_i = randn(1, 1.5*length(t)+bp_mod_fo+1);
    noise_mod_i = filter(bp_filt_mod,1,noise_mod_i);
    noise_mod_i = noise_mod_i(bp_mod_fo+1:end);
    
    noise_mod_i = max(noise_mod_i,0); %h rectify
    noise_mod_i = filter(lp_filt,1,noise_mod_i);
    noise_mod_i = noise_mod_i(length(lp_filt)+1:length(lp_filt)+length(t));
    
    noise_mod(i,:) = noise_mod_i / max(noise_mod_i);
    fprintf(1,'%d/200 \n',i)
end

save(['RandMod_Hrect_' num2str(n_mod_cuts(2)) '.mat'],'noise_mod','n_mod_cuts','fs');











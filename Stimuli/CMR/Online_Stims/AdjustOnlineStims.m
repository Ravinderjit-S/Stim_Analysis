clear
folder_loc = '/home/ravinderjit/Documents/OnlineStim_WavFiles/Old/CMR/';


mod_bands = [2 10; 36 44; 101 109];
Block =1;
exp_modband = mod_bands(2,:);
Fold_path = [folder_loc 'Mod_' num2str(exp_modband(1)) '_' num2str(exp_modband(2)) '/' 'Block' num2str(Block) '/' ];

folder_targ = ['/home/ravinderjit/Documents/OnlineStim_WavFiles/CMR/Mod_' num2str(exp_modband(1)) '_' num2str(exp_modband(2)) '/'];
%fname = ['CMR3AFCrandMod_' num2str(exp_modband(1)) '_' num2str(exp_modband(2)) '_trial_' num2str(l) '.wav'];
    
%SNRdB_1 = 0:-6:-30;
SNRdB_1 = 6:-6:-24;
SNRdB_2 = 6:-6:-24; 
ntrials = 10;

Stim_Dat = load([Fold_path 'Stim_Data.mat']);
correct = Stim_Dat.correct';
SNRdB_exp = Stim_Dat.SNRdB_exp;

for j = 1:length(SNRdB_1)
    SNR_inds_1(:,j) = find(SNRdB_exp(:,1) == SNRdB_1(j) & SNRdB_exp(:,2)==1);
end
    
for j = 1:length(SNRdB_2)
    SNR_inds_2(:,j) = find(SNRdB_exp(:,1) == SNRdB_2(j) & SNRdB_exp(:,2)==0);
end

SNR_FinalInds = [];
for j=1:10
    SNR_FinalInds = [SNR_FinalInds SNR_inds_1(j,:)];
    SNR_FinalInds = [SNR_FinalInds SNR_inds_2(j,:)];
end

correct = correct(SNR_FinalInds);
SNRdB_exp = SNRdB_exp(SNR_FinalInds,:);


for j = 1:length(SNR_FinalInds)
    fname_old = ['CMR3AFCrandMod_' num2str(exp_modband(1)) '_' num2str(exp_modband(2)) '_trial_' num2str(SNR_FinalInds(j)) '.wav'];
    fname_new = ['CMR3AFCrandMod_' num2str(exp_modband(1)) '_' num2str(exp_modband(2)) '_trial_' num2str(j) '.wav'];
    movefile([Fold_path fname_old],[folder_targ fname_new])
end
movefile([Fold_path 'volstim.wav'],[folder_targ 'volstim.wav']);
save([folder_targ 'Stim_Data.mat'],'SNRdB_exp','correct','exp_modband')









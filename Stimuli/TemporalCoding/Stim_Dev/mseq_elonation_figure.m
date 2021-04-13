fs = 48828;
upperF = 20;
m =9;
Point_len = floor(fs/upperF);

rng('default');
s = rng(0);

mseq = mls2(m,0,s);

fig_path = '/media/ravinderjit/Data_Drive/Data/Figures/DynBin/';

mseq_elong = [];
for i =1:length(mseq)
    mseq_elong = [mseq_elong ones(1,Point_len)*mseq(i)]; %#ok
end

t1 = 0:1/fs:length(mseq)/fs-1/fs;
t2 = 0:1/fs:length(mseq_elong)/fs-1/fs;

points_show = 50;

figure,
plot(t1(1:points_show),mseq(1:points_show),'color','k','linewidth',2)
ylim([-1.05,1.05])
xlim([0,t1(points_show)])
yticks([-1 0 1])
xticks([0,0.0005,.001])
xticklabels({'0','0.0005','.001'})
xlabel('Time (sec)')
box off
set(gca,'FontSize',16)
print([fig_path 'mseq_regular.svg'],'-dsvg')


figure,plot(t2(1:points_show*Point_len),mseq_elong(1:points_show*Point_len), ... 
    'color','k','linewidth',2)
ylim([-1.05,1.05])
xlim([0,t2(points_show*Point_len)])
yticks([-1 0 1])
xticks([0,1.2,2.4])
%xticklabels({'0','0.0005','.001'})
xlabel('Time (sec)')
box off
set(gca,'FontSize',16)
print([fig_path  'mseq_elongated.svg'],'-dsvg')


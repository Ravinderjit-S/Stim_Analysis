clear

load('MEMR_pressures.mat')

thresh = 0.1;

thresholds_w = zeros(length(Subjects),1);
thresholds_hp = zeros(length(Subjects),1);

for s = 1:length(Subjects)
    
    memr_w = MEMR_white{s};
    memr_hp = MEMR_hp{s};

    e_w = elict_white{s};
    e_hp = elict_hp{s};

   
   fit_w = fit(e_w',memr_w,'smoothingspline');
   fit_hp = fit(e_hp',memr_hp, 'smoothingspline');
   
          
   elic_weval = e_w(1):e_w(end);
   elic_hpeval = e_hp(1):e_hp(end);
   
   mw_eval = feval(fit_w, elic_weval);
   mhp_eval = feval(fit_hp, elic_hpeval);
   
   thresh_wind = find(mw_eval < thresh, 1, 'last');
   thresh_hpind = find(mhp_eval < thresh, 1, 'last');
   
   thresholds_w(s) = elic_weval(thresh_wind);
   thresholds_hp(s) = elic_hpeval(thresh_hpind);
   
   figure(), hold on
   plot(e_w,memr_w,'bx')
   plot(elic_weval, mw_eval,'b')
   plot(elic_weval(thresh_wind), mw_eval(thresh_wind),'rx')
   
   plot(e_hp,memr_hp,'gx')
   plot(elic_hpeval, mhp_eval,'g')
   plot(elic_hpeval(thresh_hpind), mhp_eval(thresh_hpind),'mx')
   
   title(Subjects{s})

    
end

save_loc = '/media/ravinderjit/Data_Drive/Data/EEGdata/MTB/MEMR/';
save([save_loc, 'MEMR_thresholds.mat'],'Subjects','thresholds_w','thresholds_hp')













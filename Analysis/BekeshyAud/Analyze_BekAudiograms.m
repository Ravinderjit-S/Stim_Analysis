clear
data_loc = '/media/ravinderjit/Data_Drive/Data/MTB_Behavior/Audiograms/';


Subjects = {'S072','S078','S088','S207','S246','S259','S260','S268', ...
    'S269','S270', 'S271','S272','S273','S274','S277','S279', ...
    'S280','S281','S282','S284','S285','S288','S290','S291','S303', ...
    'S305','S308','S309','S310'};

audiogram_L = zeros(14,length(Subjects));
audiogram_R = zeros(14,length(Subjects));

for i=1:length(Subjects)
    files = dir([data_loc Subjects{i}]);
    
    if length(files) < 4 %Some sujects missing Left ear
        Rfile = files(3).name;
        Raud = load([data_loc, Subjects{i}, '/', Rfile]);
        [R_fit, ~] = fitaudiogram(Raud.flist, Raud.Ltrack);
        L_fit = NaN(14,1);
    else
        Lfile = files(3).name;
        Rfile = files(4).name;
        
        Laud = load([data_loc, Subjects{i}, '/', Lfile]);
        Raud = load([data_loc, Subjects{i}, '/', Rfile]);
        
        [L_fit, f_fit] = fitaudiogram(Laud.flist, Laud.Ltrack);
        [R_fit, ~] = fitaudiogram(Raud.flist, Raud.Ltrack);
        
    end
    
    
    audiogram_L(:,i) = L_fit;
    audiogram_R(:,i) = R_fit;
    
end

save([data_loc, 'BekAud.mat'],'Subjects','f_fit','audiogram_L','audiogram_R')

    
    
    
    
    
    


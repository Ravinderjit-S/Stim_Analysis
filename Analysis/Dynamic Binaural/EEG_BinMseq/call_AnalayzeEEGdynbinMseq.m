clear

Subjects = [{'S001'},{'S132'},{'S203'},{'S204'},{'S205'},{'S206'},{'S207'},{'S208'}];
subj = Subjects{1};
Aud_channels=[5,8,9,26,31,32];
Num_noiseFloors = 50;
Keep_H = 1;
% fprintf([subj '\n'])
% AnalyzeEEG_DynBin_Mseq(subj,Aud_channels,Num_noiseFloors,Keep_H)

% Subjects = {'S001'};

for i =1:numel(Subjects)
    subj = Subjects{i};
    fprintf([subj '\n'])
    AnalyzeEEG_DynBin_Mseq(subj,Aud_channels,Num_noiseFloors,Keep_H)
end





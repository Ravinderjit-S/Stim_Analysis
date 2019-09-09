dates = [{'09.16.18'},{'09.20.18'},{'09.21.18'},{'10.01.18'},{'10.12.18'}];
Add_dummies = 100;
for i = 1:numel(dates)
    AddNFMSO(dates{i}, Add_dummies); %adds 25 NF
end

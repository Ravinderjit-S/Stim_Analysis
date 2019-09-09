function [data] = ExtractData(Pic_nums, Dataname,directory)
%extracts pictures from nel data and turns it into cell structure called
%data

    cd(directory)
    data = cell(1,Pic_nums(end));
    for i = Pic_nums
        [pic fname] = loadPic2(i);
        if exist([fname '_ADdata.mat']) == 2
            load([fname '_ADdata.mat'])
            pic.ADdata.RawTrace = ADdata;
            pic.ADdata.triggerSamples = triggerSamples;
        end
        data{i} = pic;
    end
    save(Dataname,'data')
end

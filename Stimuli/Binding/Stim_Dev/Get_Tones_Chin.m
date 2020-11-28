function [Tones_f, ERBspace] = Get_Tones_Chin(Tones_num, ERB_spacing, f_start, f_end)
%This function is designed to return a set of tones b/t f_start and f_end
%in a manner to have ERB spacing specified by ERB_spacing. If no
%ERB_spacing is indicated then tones will be spaced with equal ERB spacing
    if isempty(ERB_spacing)
        Tot_ERBs = ERBs_chin(f_end) - ERBs_chin(f_start);
        ERBspace = Tot_ERBs / (Tones_num-1); %ERBs b/t tones
        curr_ERBs = ERBs_chin(f_start);
        for i = 1:Tones_num
            Tones_f(i) = FreqERBs_chin(curr_ERBs);
            curr_ERBs = curr_ERBs + ERBspace;
        end
    else
        ERBspace = ERB_spacing;
        curr_ERBs = ERBs_chin(f_start);
        j = 1;
        ERB_max = ERBs_chin(f_end);
        goon = true;
        while goon
            Tones_f(j) = FreqERBs_chin(curr_ERBs);
            curr_ERBs = curr_ERBs + ERBspace;
            if curr_ERBs > ERB_max
                goon = false;
            end
            j = j+1;
        end
    end
end

function [erbs] = ERBs_chin(f)
%formula must take in kHz but hz is returned
    f = f/1000; 
    erbs = (1/0.14)*f^0.4;
end

function [freq] = FreqERBs_chin(ERBs)
%This function will return the frequency for a particular ERBs in hz.
%Formula uses khz but I convert to hz before returning 
    freq = (0.14*ERBs)^(1/0.4);
    freq = freq*1000; %convert to Hz
end
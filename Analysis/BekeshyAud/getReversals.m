function [upList, downList, revList] = getReversals(x)
nReversals = 0;
downList = [];
upList = [];
revList = [];
for k = 3:numel(x)
    if((x(k-1) > x(k)) && (x(k-1) > x(k-2)))
        nReversals = nReversals + 1;  revList = [revList, (k-1)];
        downList = [downList, (k-1)];
    end
    if((x(k-1) < x(k)) && (x(k-1) < x(k-2)))
        nReversals = nReversals + 1;  revList = [revList, (k-1)];
        upList = [upList, (k-1)]; %#ok<*AGROW>
    end
end
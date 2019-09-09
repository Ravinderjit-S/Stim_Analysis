function [] = AudiogramWelcome(PS,buttonBox,textlocH,textlocV,line2line)
%%This function just contains the code to welcome subject to the experiment.
    
    info = strcat('We will begin with an Audiogram!');
    %info = strcat('This is block #',blockNumStr,'/',totalBlocks,'...');
    Screen('DrawText',PS.window,info,textlocH,textlocV,PS.white);
    info = strcat('When you are ready: Press any button twice to begin...');
    Screen('DrawText',PS.window,info,textlocH,textlocV+line2line,PS.white);
    Screen('Flip',PS.window);

    if buttonBox  %Subject pushes button twice to begin
        getResponse(PS.RP);
        getResponse(PS.RP);
    else
        getResponseKb; %#ok<UNRCH>
        getResponseKb;
    end

end
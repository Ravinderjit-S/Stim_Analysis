function [resp] = GetResponse_Feedback(PS,feedback, feedbackDuration,buttonBox,correct_ans)
%This function will get the subjects response and provide feedback  
%feedback = 1 or 0 depending on if you want to provide feedback


    renderVisFrame(PS,'RESP');
    Screen('Flip',PS.window);

    if(buttonBox)
        resp = getResponse(PS.RP);
        if numel(resp)>=1  %accounting for multiple button presses
            resp = resp(1);
        end
    else
        resp = getResponseKb; %#ok<UNRCH>
    end
    
    if resp == correct_ans 
        correct_fb = 1; 
    else
        correct_fb = 0;
    end

    if(feedback)
        if(correct_fb)
            renderVisFrame(PS,'GO');
        else
            renderVisFrame(PS,'NOGO');
        end
    end
    Screen('Flip',PS.window);
    WaitSecs(feedbackDuration);


end
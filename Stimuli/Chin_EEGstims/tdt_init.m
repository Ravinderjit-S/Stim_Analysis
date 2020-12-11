function [RP1, RP2, RX8, PA5_1,PA5_2,PA5_3,PA5_4,tdt_info] = tdt_init(fig_num)
%Connect to TDT outside of NEL to play auditory sitmuli (not recording
%anything with this setup)
% based on tdtinit.m in NEL code
%R.S. 10/31/2020

%%

f1=figure(fig_num);
fig_loc = [0 0 1 1];
set(f1,'Position',fig_loc,'visible','off')
RP = actxcontrol('RPco.x',fig_loc,f1);

yesUSB= invoke(RP,'ConnectRP2', 'USB', 1);
yesGB= invoke(RP,'ConnectRP2', 'GB', 1);

if yesUSB && ~yesGB
    tdt_info.TDTcommMode= 'USB';
elseif yesGB && ~yesUSB
    tdt_info.TDTcommMode= 'GB';
end

% Assuming RP2s only: will be different for RX8
% Check how many RP2s

RP1 = actxcontrol('RPco.x',fig_loc);
status(1) = RP1.ConnectRP2(tdt_info.TDTcommMode,1);

RP2 = actxcontrol('RPco.x',fig_loc);
status(2) = RP2.ConnectRP2(tdt_info.TDTcommMode,2);

RX8 = actxcontrol('RPco.x',fig_loc);
status(3) = RX8.ConnectRX8(tdt_info.TDTcommMode,1);

PA5_1 = actxcontrol('PA5.x',fig_loc);
status(4) = PA5_1.ConnectPA5(tdt_info.TDTcommMode,1);

PA5_2 = actxcontrol('PA5.x',fig_loc);
status(5) = PA5_2.ConnectPA5(tdt_info.TDTcommMode,2);

PA5_3 = actxcontrol('PA5.x',fig_loc);
status(6) = PA5_3.ConnectPA5(tdt_info.TDTcommMode,3);

PA5_4 = actxcontrol('PA5.x',fig_loc);
status(7) = PA5_4.ConnectPA5(tdt_info.TDTcommMode,4);

tdt_info.status=status;

if ~all(status)
    error('Something not connected')
end




function [] = plot2yaxis(x1,x2,y1,y2)

figure
yyaxis left
plot(x1,y1,'o')
ylabel('

yyaxis right
plot(x2,y2,'^')


end



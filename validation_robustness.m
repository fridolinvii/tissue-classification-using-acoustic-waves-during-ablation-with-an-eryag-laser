clear all, close all

Str{1} = "plot_csv/FC/validation_frequency_";
Str{2} = "plot_csv/FC/validation_time_";
Str{3} = "plot_csv/CNN/validation_frequency_";
Str{4} = "plot_csv/CNN/validation_time_";
Str{5} = "plot_csv/RNN/validation_frequency_";
Str{6} = "plot_csv/RNN/validation_time_";

Title{1} = "FC Frequency";
Title{2} = "FC Time";
Title{3} = "CNN Frequency";
Title{4} = "CNN Time";
Title{5} = "RNN Frequency";
Title{6} = "RNN Time";


cc = 0;

TITLE = [];
modd{1} = [];
modd{2} = [];
modd{3} = [];
modd{4} = [];
modd{5} = [];


xyFont = 22;
xFont = 26;
yFont = 26;


for k = 1:6
    
    
    out = 0;
    for ii = 0:4
        out = out + load(Str{k}+num2str(ii)+'.out');
    end
    out = out/5;
    TITLE = [TITLE,Title{k}];
    
    for i = unique(out(:,1))'
        cc = mod(cc,10)+1;
        
        %    subplot(length(unique(out(:,1))),1,i)
        
        
        if ~isempty(strfind(Str{k},'FC'))
            figure(1)
        elseif ~isempty(strfind(Str{k},'CNN'))
            figure(2)
        else
            figure(3)
        end
        
        
        subplot(2,5,cc)
        
        Out = out(out(:,1)==i,2:end);
        x = Out(:,1);
        y = Out(:,2);
        z = Out(:,3);
        
        x0 = reshape(x,length(unique(y)),length(unique(x)));
        y0 = reshape(y,length(unique(y)),length(unique(x)));
        z0 = reshape(z,length(unique(y)),length(unique(x)));
        
        surf(x0,y0,z0)
        view([0,0,1])
        %title(Title{k})
        title(mean(z))
        axis('tight')
        xlabel('Time Shift','FontSize', xFont)
        ylabel('Magnitude Variation','FontSize', yFont)
        set(gca,'FontSize',xyFont)
        modd{i} =  [modd{i};mean(z)];
        %  sum(z)
        
       % colorbar
caxis([0.2 1])
        min(z0(:))
        
    end
end

% 
% figure(3)
% subplot(1,2,1)
% surf(x0,y0,z0)
% colorbar
% caxis([0.2 1])
% view([0,0,1])
% %title(Title{k})
% title(mean(z))
% axis('tight')
% xlabel('Time Shift','FontSize', xFont-10)
% ylabel('Magnitude Variation','FontSize', yFont-10)
% set(gca,'FontSize',xyFont-10)
% modd{i} =  [modd{i};mean(z)];
% colorbar('southoutside')
% 
% 
% T = table(TITLE',modd{1} ,modd{2},modd{3},modd{4},modd{5});
% T.Properties.VariableNames = {'Title','0.1-0.8Mhz','0.115-0.27Mhz','0.27-0.53Mhz','0.53-0.8Mhz','0.1-0.37Mhz'}
% 
% pause(1)
% figure
% 
% load('M.mat')
% subplot(2,3,1)
% plot(M(500:1499))
% xlabel('Time Frame')
% ylabel('Magnitude')
% title('Original')
% 
% subplot(2,3,2)
% plot(M(150+(500:1499)))
% xlabel('Time Frame')
% ylabel('Magnitude')
% title('Time Shift +150')
% 
% subplot(2,3,3)
% plot(M(-150+(500:1499)))
% xlabel('Time Frame')
% ylabel('Magnitude')
% title('Time Shift -150')
% 
% subplot(2,3,4)
% I = 500:1499;
% plot(M(I).*(1+1*exp(-abs(M(I)))))
% xlabel('Time Frame')
% ylabel('Magnitude')
% title('P*(1+1*exp(-abs(P)))')
% 
% subplot(2,3,5)
% I = 500:1499;
% plot(M(I).*(1-1*exp(-abs(M(I)))))
% xlabel('Time Frame')
% ylabel('Magnitude')
% title('P*(1-1*exp(-abs(P)))')
% 
% 
% subplot(2,3,6)
% I = (500:1499)-150;
% plot(M(I).*(1-0.5*exp(-abs(M(I)))))
% xlabel('Time Frame')
% ylabel('Magnitude')
% title('P*(1-0.5*exp(-abs(P)))')
% 




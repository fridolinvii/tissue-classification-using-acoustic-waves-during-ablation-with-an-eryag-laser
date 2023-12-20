clear all, close all;


% lineFont = 5;
% xyFont = 18;
% xFont = 20;


lineFont = 5;
xyFont = 22;
xFont = 26;
yFont = 26;

load colorblind_colormap
colorblind = colorblind([1 2 4 6 8 11],:);


for o = 1:3

Ord{3} = 'RNN/';
Ord{2} = 'Conv_1Layer/';
Ord{1} = 'SET_small_FC/';

if o==3
    figure(2)
subplot(2,2,2*(o-2)-1)
else

subplot(2,2,2*o-1)

end
NN = 200;
error = zeros(6,NN);



eeFreq = [];
mm = 6;
for n= [1,2,5,3,4]
    ERROR = [];
    for ii = 0:4
        SET = ['SET_small_',num2str(ii),'_FC'];
        for e = 0:NN-1
            disp([n,ii,e,NN])
            
            filename = [Ord{o},SET,'/data/ErYAG/epoch_frequency_length_',num2str(n),'/epoch',num2str(e),'.txt'];
            filename2 = [Ord{o},SET,'/data/ErYAG/epoch_frequency_length_',num2str(n),'/epoch',num2str(e+1),'.txt'];
            
            if isfile(filename2)
                delimiterIn = ' ';
                headerlinesIn = 2;
                err = importdata(filename,delimiterIn,headerlinesIn);
                err = str2double(err{2}(44:end-2));
                error(n,e+1) = err;
                if err == 100
                    eeFreq = [eeFreq,e];
                    break
                end
            end
            
        end
        ERROR = [ERROR;error(n,:)];
    end
    
    ERROR = ERROR(:,sum(ERROR)>0);
    ERROR(ERROR==0) = 100;
    ERROR = mean(ERROR,1);
    
    hold on
    if n==3
        plot(ERROR,'Color',colorblind(4,:),'LineWidth', lineFont)
    elseif n == 5
        plot(ERROR,'Color',colorblind(3,:),'LineWidth', lineFont)
    elseif n == 4
        plot(ERROR,'Color',colorblind(5,:),'LineWidth', lineFont)
    else
        plot(ERROR,'Color',colorblind(n,:),'LineWidth', lineFont)
    end
end


set(gca,'FontSize',xyFont)
xlabel('number of epoch','FontSize', xFont)
ylabel('accuracy [%]','FontSize', xFont)
axis('tight')
legend('0.100-0.80 Mhz','0.115-0.27 Mhz','0.100-0.37 Mhz','0.270-0.53 Mhz','0.530-0.80 Mhz','FontSize', xFont)


if o==3
   subplot(2,2,2*(o-2)) 
else
    subplot(2,2,2*o)
end
NN = 43;
error = zeros(6,NN);
eeTime = [];
mm = 6;
for n= [1 2 5 3 4] %1:4 %[5 6 4 3 2 1]
    ERROR = [];
    for ii = 0:4
        SET = ['SET_small_',num2str(ii),'_FC'];
        for e = 0:NN-1
            disp([n,ii,e,NN])
            filename = [Ord{o},SET,'/data/ErYAG/epoch_time_length_',num2str(n),'/epoch',num2str(e),'.txt'];
            filename2 = [Ord{o},SET,'/data/ErYAG/epoch_time_length_',num2str(n),'/epoch',num2str(e+1),'.txt'];
            
            if isfile(filename2)
                delimiterIn = ' ';
                headerlinesIn = 2;
                err = importdata(filename,delimiterIn,headerlinesIn);
                err = str2double(err{2}(44:end-2));
                error(n,e+1) = err;
                if err == 100
                    eeTime = [eeTime,e];
                    break
                end
            end
            
        end
        ERROR = [ERROR;error(n,:)];
    end
    
    ERROR = ERROR(:,sum(ERROR)>0);
    ERROR(ERROR==0) = 100;
    ERROR = mean(ERROR,1);
    
    hold on
    if n==3
        plot(ERROR,'Color',colorblind(4,:),'LineWidth', lineFont)
    elseif n == 5
        plot(ERROR,'Color',colorblind(3,:),'LineWidth', lineFont)
    elseif n == 4
        plot(ERROR,'Color',colorblind(5,:),'LineWidth', lineFont)
    else
        plot(ERROR,'Color',colorblind(n,:),'LineWidth', lineFont)
    end

    
end


set(gca,'FontSize',xyFont)
xlabel('number of epoch','FontSize', xFont)
ylabel('accuracy [%]','FontSize', xFont)
axis('tight')
legend('0.100-0.80 Mhz','0.115-0.27 Mhz','0.100-0.37 Mhz','0.270-0.53 Mhz','0.530-0.80 Mhz','FontSize', xFont)

end

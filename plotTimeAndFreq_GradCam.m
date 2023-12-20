
clear all, close all

load colorblind_colormap
colorblind = colorblind([1 2 4 6 8 11],:);


LABEL{1} = 'Soft Bone';
LABEL{2} = 'Skin';
LABEL{3} = 'Muscle';
LABEL{4} = 'Hard Bone';
LABEL{5} = 'Fat';


lab = 4;
count = 0;
LL_load = [1,2,5,3,4];
LLL = 0:5;

for m = 1:5
    
    
    len = num2str(LL_load(m));
    
    image_true = table2array(readtable(['plot_csv/',len,'_image_true.csv']));
    image_freq = table2array(readtable(['plot_csv/',len,'_image_freq.csv']));
    image_time = table2array(readtable(['plot_csv/',len,'_image_time.csv']));
    time = load(['plot_csv/',len,'_time.csv'])*1e3;
    
    
    freq = load(['plot_csv/',len,'_freq.csv'])*1e-6;
    
    
    % tissue ('SoftBone', 'Skin', 'Muscle', 'HardBone', 'Fat')
    % classes = ('HardBone','SoftBone','Fat','Skin','Muscle')
    
    %     if isequal(len,'0')
    %         freq = [0;table2array(readtable(['plot_csv/',len,'_freq.csv']))]*1e-6;
    %         freq = freq(1:1001);
    %         image_freq = image_freq(1:1001,l);
    %     else
    %         freq = [table2array(readtable(['plot_csv/',len,'_freq.csv']))]*1e-6;
    %         freq = [freq];
    %     end
    %line = {'-','--','-.',':','--'};
    %     line = {'b-','r-','darkgreen-','c-','m-'};
    
    n = 1;
    nf = 1;
    
    %     lwFont = 2;
    %     xyFont = 15;
    %     xFont = 18;
    %     yFont = 18;
    lwFont = 4;
    xyFont = 22;
    xFont = 26;
    yFont = 26;
    
    ll = 0;
    llf = 0;
    
    
    
    for l = lab
        

        if count < 6
            figure(1)
            
            subplot(3,2,count+2)
            set(gca,'FontSize',xyFont)
            hold on
            for n = 1:LLL(m)
                lineH = plot(1,1);
                color = get(lineH, 'Color');
                delete(lineH);
            end
            plot(time,image_time(:,l),'Color',colorblind(m,:),'LineWidth',lwFont)
            time_ticks(time)
            axis('tight')
            ylim([-1,1])
            xlabel('time [ms]','FontSize', xFont)
            ylabel('freq. filter P. var.','FontSize', yFont)
            
            
            
            
            subplot(3,2,count+1)
            set(gca,'FontSize',xyFont)
            hold on
            for n = 1:LLL(m)
                lineH = plot(1,1);
                color = get(lineH, 'Color');
                delete(lineH);
            end
            b = bar(freq,image_freq(:,l),'LineWidth',lwFont/lwFont)
            b.FaceColor = colorblind(m,:);
            %b.EdgeColor = colorblind(m,:);
            axis('tight')
            xlabel('frequency [Mhz]','FontSize', xFont)
            ylabel('abs. P. var.','FontSize', yFont)
            ylim([0,max(image_freq(:))])
            freq_x_ticks(freq,m)
            %         count = 1+count;
            count = count + 2;
        else
            figure(2)
            
            
            subplot(3,2,count+2-6)
            set(gca,'FontSize',xyFont)
            hold on
            for n = 1:LLL(m)
                lineH = plot(1,1);
                color = get(lineH, 'Color');
                delete(lineH);
            end
            plot(time,image_time(:,l),'Color',colorblind(m,:),'LineWidth',lwFont)
            time_ticks(time)
            axis('tight')
            ylim([-1,1])
            xlabel('time [ms]','FontSize', xFont)
            ylabel('freq. filter P. var.','FontSize', yFont)
            
            subplot(3,2,count+1-6)
            set(gca,'FontSize',xyFont)
            hold on
            for n = 1:LLL(m)
                lineH = plot(1,1);
                color = get(lineH, 'Color');
                delete(lineH);
            end
            b = bar(freq,image_freq(:,l),'LineWidth',lwFont/lwFont)
            b.FaceColor = colorblind(m,:);
            %b.EdgeColor = colorblind(m,:);
            axis('tight')
            xlabel('frequency [Mhz]','FontSize', xFont)
            ylabel('abs. P. var.','FontSize', yFont)
            ylim([0,max(image_freq(:))])
            
            freq_x_ticks(freq,m)
            %         count = 1+count;
            count = count + 2;
        end
        
        
        
    end
    
    
end


function freq_x_ticks(freq,m)
if m == 1
    a1 = .25;
    a2 = 0.2;
    a3 = 0.79;
    xticks([0.1, a1:a2:a3, freq(end)])
    xticklabels({0.1, a1:a2:a3, 0.8})
   % xticklabels({0.1:0.1:0.7 0.8})
elseif m == 2
    xticks([freq(1), 0.17:0.05:0.24, freq(end)])
    xticklabels({0.115, 0.17:0.05:0.24, 0.27})
elseif m == 4
    a1 = .35;
    a2 = 2*0.04;
    a3 = 0.44;
    xticks([freq(1), a1:a2:a3, freq(end)])
    xticklabels({0.27, a1:a2:a3, 0.53})
elseif m == 5
    a1 = .6;
    a2 = 2*0.03;
    a3 = 0.73;
    xticks([freq(1), a1:a2:a3, freq(end)])
    xticklabels({0.53, a1:a2:a3, 0.8})
elseif m == 3
    a1 = .17;
    a2 = 2*0.03;
    a3 = 0.3;
    xticks([freq(1), a1:a2:a3, freq(end)])
    xticklabels({0.1, a1:a2:a3, 0.37})
end
end

function time_ticks(time)
    xticks([time(1),0.02:0.02:0.08, time(end)])
    xticklabels({0.0:0.02:0.1})
end

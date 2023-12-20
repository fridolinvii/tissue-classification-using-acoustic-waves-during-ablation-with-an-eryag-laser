
clear all, close all

load colorblind_colormap
colorblind = colorblind([1 2 4 6 8 11],:);


len = '1';

image_true = table2array(readtable(['plot_csv/',len,'_image_true.csv']));
image_freq = table2array(readtable(['plot_csv/',len,'_image_freq.csv']));
image_time = table2array(readtable(['plot_csv/',len,'_image_time.csv']));
time = [0;table2array(readtable(['plot_csv/',len,'_time.csv']))]*1e3;


% tissue ('SoftBone', 'Skin', 'Muscle', 'HardBone', 'Fat')
% classes = ('HardBone','SoftBone','Fat','Skin','Muscle')

LABEL{1} = 'Soft Bone';
LABEL{2} = 'Skin';
LABEL{3} = 'Muscle';
LABEL{4} = 'Hard Bone';
LABEL{5} = 'Fat';




if isequal(len,'0')
    freq = [0;table2array(readtable(['plot_csv/',len,'_freq.csv']))]*1e-6;
    freq = freq(1:1001);
    image_freq = image_freq(1:1001,:);
else
    freq = [table2array(readtable(['plot_csv/',len,'_freq.csv']))]*1e-6;
    freq = [0.1;freq];
end
%line = {'-','--','-.',':','--'};
line = {'r-','b-','g-','c-','m-'};
color = 0.25*[1,1,1];

n = 1;
nf = 1;

lwFont = 2;
xyFont = 15;
xFont = 20;
yFont = 20;

lwFont = 5;
xyFont = 22;
xFont = 26;
yFont = 26;


ll = 0;
llf = 0;

L = [2:5,1];
count = 0;
for l = L
    
    %
    %     subplot(1,3,1)
    %     hold on
    %     plot(time(1:n:end),image_true(1:n:end,l),line{l},'LineWidth',lwFont)
    %     subplot(1,3,2)
    %     hold on
    %     plot(time(1:n:end),image_time(1:n:end,l),line{l},'LineWidth',lwFont)
    %     subplot(1,3,3)
    %     hold on
    %     plot(freq(1:nf:end),image_freq(1:nf:end,l),line{l},'LineWidth',lwFont)
    
    
    
    LABEL{l}
    subplot(5,3,count+1)
    set(gca,'FontSize',xyFont)
    hold on
    plot(time,image_true(:,l),'Color',color,'LineWidth',lwFont)
    axis('tight')
    ylim([min(image_true(:)),max(image_true(:))])
    xlabel('time [ms]','FontSize', xFont)
    ylabel('P. var.','FontSize', yFont)
    xticks([0. 0.05 0.099999])
    xticklabels({0 0.05 0.1})
    
    
    
    
    subplot(5,3,count+2)
    set(gca,'FontSize',xyFont)
    hold on
    plot(time,image_time(:,l),'Color',color,'LineWidth',lwFont)
    axis('tight')
    ylim([-1,1])
    xlabel('time [ms]','FontSize', xFont)
    ylabel('norm. P. var.','FontSize', yFont)
        xticks([0. 0.05 0.099999])
    xticklabels({0 0.05 0.1})
    
    
    
    
    subplot(5,3,count+3)
    set(gca,'FontSize',xyFont)
    hold on
    b = bar(freq,image_freq(:,l),'LineWidth',lwFont-2);
    b.FaceColor = color;
    b.EdgeColor = color;
    axis('tight')
    xlabel('frequency [Mhz]','FontSize', xFont)
    ylim([0,max(image_freq(:))])
    ylabel('abs. P. var.','FontSize', yFont)

    count = 3+count;
    
end



% subplot(3,5,1)
% ylabel('pressure variation','FontSize', yFont)
% subplot(3,5,6)
% ylabel('norm. pressure variation','FontSize', yFont)
% subplot(3,5,11)
% ylabel('abs. pressure variation','FontSize', yFont)


% for l = L
%     subplot(3,5,l)
%     title(LABEL{l},'FontSize', yFont)
% end


%
% xFont = 20;
% legendFont = 18;
% xyFont = 18;
%
%
% subplot(2,3,1)
% set(gca,'FontSize',xyFont)
% legend(LABEL{2},LABEL{3},LABEL{4},LABEL{5},LABEL{1},'FontSize', legendFont)
% xlabel('time [ms]','FontSize', xFont)
% axis('tight')
% subplot(2,3,2)
% set(gca,'FontSize',xyFont)
% legend(LABEL{2},LABEL{3},LABEL{4},LABEL{5},LABEL{1},'FontSize', legendFont)
% xlabel('time [ms]','FontSize', xFont)
% axis('tight')
% subplot(2,3,3)
% set(gca,'FontSize',xyFont)
% legend(LABEL{2},LABEL{3},LABEL{4},LABEL{5},LABEL{1},'FontSize', legendFont)
% xlabel('time [ms]','FontSize', xFont)
% axis('tight')
%
% subplot(2,3,4)
% set(gca,'FontSize',xyFont)
% legend(LABEL{2},LABEL{3},LABEL{4},LABEL{5},LABEL{1},'FontSize', legendFont)
% xlabel('time [ms]','FontSize', xFont)
% axis('tight')
% subplot(2,3,5)
% set(gca,'FontSize',xyFont)
% legend(LABEL{2},LABEL{3},LABEL{4},LABEL{5},LABEL{1},'FontSize', legendFont)
% xlabel('time [ms]','FontSize', xFont)
% axis('tight')
% subplot(2,3,6)
% set(gca,'FontSize',xyFont)
% legend(LABEL{2},LABEL{3},LABEL{4},LABEL{5},LABEL{1},'FontSize', legendFont)
% xlabel('frequency [Mhz]','FontSize', xFont)
% axis('tight')

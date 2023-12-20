clear all, close all

lineFont = 5;
xyFont = 22;
xFont = 26;
yFont = 26;



load colorblind_colormap
colorblind = colorblind([1 2 4 6 8 11],:);

len = 1;
subplot(1,2,1)
RANGE = [];
for ii = 0:4
    SET = ['SET_small_',num2str(ii),'_FC/paper/'];


range0 = load([SET,'train_normalized/range0_eryag_frequency_length_',num2str(len),'.csv']);
range1 = load([SET,'train_normalized/range1_eryag_frequency_length_',num2str(len),'.csv']);
range2 = load([SET,'train_normalized/range2_eryag_frequency_length_',num2str(len),'.csv']);
range3 = load([SET,'train_normalized/range3_eryag_frequency_length_',num2str(len),'.csv']);
range4 = load([SET,'train_normalized/range4_eryag_frequency_length_',num2str(len),'.csv']);


% image_freq = table2array(readtable('image_freq.csv'));
% freq = image_freq(:,6);

freq = [table2array(readtable(['SET_small_',num2str(0),'_FC/paper/','plot_csv/1_freq.csv']))]*1e-6;
freq(1) = 0.1; freq(end) = 0.8;
% freq = [0.1;freq];

freq_conv = linspace(min(freq),max(freq),length(range0));
range = ([range0, range1, range2, range3, range4]); %./max([range0, range1, range2, range3, range4]);
range = sum(range,2);
%range = mean(range,2);
range = interp1(freq_conv,range,freq);


% range0 = interp1(freq_conv,range0,freq);
% range1 = interp1(freq_conv,range1,freq);
% range2 = interp1(freq_conv,range2,freq);
% range3 = interp1(freq_conv,range3,freq);
% range4 = interp1(freq_conv,range4,freq);



%  range = range-min(range);
  range = range/max(range);

RANGE = [RANGE,range];

end

range = mean(RANGE,2);

ff_0 = [min(freq), 0.115, 0.27, 0.53]; %[0.8,0.45,0.35,.25,.2,.15];
ff_1 = [0.8, 0.27, 0.53, 0.8];
frange = 0:0.025:1;
frange = [0 0.025 0.025 0.025];
o = ones(2,1);

% colormap(colorblind)

hold on
length(freq)
plot(freq,range,'-black','LineWidth', lineFont)
for l = 1:length(ff_0) %:-1:1
    plot([ff_0(l),ff_1(l)],[frange(l),frange(l)],'-','Color',colorblind(l,:),'LineWidth', lineFont+5)
%    plot([ff_0(l),ff_0(l)],[frange(l),1],'--','Color',colorblind(l,:),'LineWidth', lineFont)
%    plot([ff_1(l),ff_1(l)],[frange(l),1],':','Color',colorblind(l,:),'LineWidth', lineFont)

end


set(gca,'FontSize',xyFont)
xlabel('frequency [Mhz]','FontSize', xFont)
ylabel('normalized activity map','FontSize', xFont)
axis('tight')
%legend('Activity Map (Grad-CAM)','0.1-0.15 Mhz','0.1-0.20 Mhz','0.1-0.25 Mhz','0.1-0.35 Mhz','0.1-0.45 Mhz','0.1-0.80 Mhz','FontSize', xFont)
legend('Activity Map (Grad-CAM)','0.100-0.800 Mhz','0.115-0.25 Mhz','0.115-0.250 Mhz','0.250-0.530 Mhz','0.530-0.800 Mhz','FontSize', xFont)


disp("0.1-0.8Mhz")
sum(range)
sum(range)/length(range)

disp("0.1-0.36")
FREQ = ( (freq>=0.1).*(freq<=0.37))>0;
sum(range(FREQ))
sum(range(FREQ)/length(range(FREQ)))


disp("0.115-0.27")
FREQ = ( (freq>=0.115).*(freq<=0.27))>0;
sum(range(FREQ))
sum(range(FREQ)/length(range(FREQ)))

disp("0.27-0.53")
FREQ = ( (freq>=0.27).*(freq<=0.53))>0;
sum(range(FREQ))
sum(range(FREQ)/length(range(FREQ)))

disp("0.53-0.8")
FREQ = ( (freq>=0.53).*(freq<=0.8))>0;
sum(range(FREQ))
sum(range(FREQ)/length(range(FREQ)))




%cum_range = cumsum(range/sum(range));
%cum_range(floor(length(cum_range)/2))

% % range = [range0/sum(range0), range1/sum(range1), range2/sum(range2), range3/sum(range3), range4/sum(range4)];
% Str = {'Hard Bone','Soft Bone','Fat','Skin','Muscle', 'Mean Grad-Cam'};
%
% lwFont = 4;
% lwFont2 = 12;
%
%
%
%
% freqRange = [0.1 0.45]; %in Mhz
% freqLength = sum((freq>=freqRange(1)).*(freq<=freqRange(2)));
%
% cum_range = cumsum(range);
% L = length(cum_range);
%
%
%
%
%
%
% for n = 1:L-freqLength
%     ff(n) = cum_range(freqLength+n)-cum_range(n);
% end
%
% o = find(max(ff)==ff);
% disp(['Frequency Range  = [',num2str(freq(o)),',',num2str(freq(o+freqLength)) ,']',])
%
% plot(range)
% cum_range(floor(length(cum_range)/2))


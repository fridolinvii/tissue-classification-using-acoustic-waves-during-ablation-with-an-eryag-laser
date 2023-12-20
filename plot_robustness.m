len = num2str(1);

lwFont = 4;
xyFont = 22;
xFont = 26;
yFont = 26;

pos = "../Tissue_Differentiation_Microphone_25.07.2019/data/ErYAG/time/MicrophoneHB_Specimens1_1000.txt";


image_true = load(pos);
image_true = image_true/max(abs(image_true));
time = (0:length(image_true)-1)*1e-7;

I = 500:1500;

time = time(1:1001)*1000;

color = 0.25*[1,1,1];

figure(2)
subplot(3,2,1)
plot(time,image_true(I),'Color',color,'LineWidth',lwFont)
xlabel('time [ms]','FontSize', xFont)
ylabel('P. var.','FontSize', yFont)
set(gca,'FontSize',xyFont)
axis([0,0.1,-1.4,1.4])

figure(1)
subplot(3,2,1)
plot(time,image_true(I-150),'Color',color,'LineWidth',lwFont)
axis('tight')
subplot(3,2,2)
plot(time,image_true(I+150),'Color',color,'LineWidth',lwFont)
axis('tight')
subplot(3,2,3)
plot(time,f(image_true(I),-1),'Color',color,'LineWidth',lwFont)
axis('tight')
subplot(3,2,4)
plot(time,f(image_true(I),1),'Color',color,'LineWidth',lwFont)
axis('tight')
subplot(3,2,5)
plot(time,f(image_true(I-150),-1),'Color',color,'LineWidth',lwFont)
axis('tight')
subplot(3,2,6)
plot(time,f(image_true(I+150),1),'Color',color,'LineWidth',lwFont)
axis('tight')


for i = 1:6
    subplot(3,2,i)
    xlabel('time [ms]','FontSize', xFont)
    ylabel('P. var.','FontSize', yFont)
    set(gca,'FontSize',xyFont)
    axis([0,0.1,-1.4,1.4])
end




function y = f(x,beta)

y = x.*(1+beta*exp(-abs(x)));

end
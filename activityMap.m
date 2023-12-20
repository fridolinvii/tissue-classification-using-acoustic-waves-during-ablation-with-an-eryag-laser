clear all, close all

a = 0;
input = 0;
for i =0:4
    Str = "CNN_fft/CNN_fft_"+num2str(i)+"_trans1_batchnorm_MultiInput2/results/gradcam/";
    activity_map = load(Str+"activity_map.txt");
    label_map = load(Str+"label_map.txt");
    spec = load(Str+"spec_map.txt");
    oo = label_map(:,1)==label_map(:,2);
    
    for ll = 0:4
        l = ll+1;
        o = label_map(:,1) == ll;
        if i == 0
            al{l} = 0;
            inputl{l} = 0;
        end
        al{l} = al{l}+sum(activity_map((o.*oo)>0,:));
        inputl{l} = inputl{l} + sum(spec((o.*oo)>0,:));
    end


    a = a+sum(activity_map(oo,:));
    input = input + sum(spec(oo,:));


end


    f = linspace(0,1e6,length(a))*1e-6;
%     subplot(2,1,1)
    subplot(2,3,6)
    an = a;
    a = a-min(a);
    plot(f,a/max(a),'red',f,input/max(input),'blue')
    xline([0.33,0.66])
    axis("tight")
%     subplot(2,1,2)
%     plot(f,input)
%     xline([0.33,0.66])
%     axis("tight")

    sum(a( ((f<0.33))>0))
    sum(a( ((f>0.33).*(f<0.66))>0))
    sum(a( ((f>0.66))>0))

    for i = 1:5
        subplot(2,3,i)
        al{i} = al{i}-min(al{i});
        plot(f,al{i}/max(al{i}),'red',f,inputl{i}/max(inputl{i}),'blue')
    end





lw = 2.5;
FontSize = 18+5;
lFontSize = 15+1;
gcaFont = 20;

%     figure(2)
%     hold on
close all
hold on
    semilogy(f,an/max(an),'b','LineWidth',lw)
    aa = 0.15;
    semilogy([0,0.3],[aa,aa], 'g',[0.4,0.6],[aa,aa], 'c',[0.7,1],[aa,aa], 'y','LineWidth',2*lw)
    set(gca,'FontSize',15,'yscale','log')
    %axis([0,1,0.1,1])
    xlabel("Frequency [MHz]",'FontSize',FontSize)
    ylabel("Normalized Activatiy Map",'FontSize',FontSize)
    %xline([0.333,0.666])
    



    count = 0;
    all = an/max(an);
    aa = aa-0.01;
for ff = f
    count = 1+count;
    if ff<0.333
        semilogy([ff,ff],[aa,all(count)],'g','LineWidth',lw)
    elseif ff<0.666
        semilogy([ff,ff],[aa,all(count)],'c','LineWidth',lw)    
    else
        semilogy([ff,ff],[aa,all(count)],'y','LineWidth',lw)    
    end
end
semilogy(f,an/max(an),'b','LineWidth',lw)
legend("Activity Map (Grad-Cam)", "Low-Freq. (0-0.333 MHz)", "Mid-Freq. (0.333-0.666 MHz)", "High-Freq. (0.666-1 MHz)",'FontSize',lFontSize)

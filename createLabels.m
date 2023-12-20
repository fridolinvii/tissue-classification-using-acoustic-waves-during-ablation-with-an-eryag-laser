clear all, close all

% load t1
Str{1} = "MicrophoneSB_Specimens";
Str{2} = "MicrophoneSkin_Specimens";
Str{3} = "MicrophoneMuscle_Specimens";
Str{4} = "MicrophoneHB_Specimens";
Str{5} = "MicrophoneFat_Specimens";



LABEL{1} = 'SoftBone';
LABEL{2} = 'Skin';
LABEL{3} = 'Muscle';
LABEL{4} = 'HardBone';
LABEL{5} = 'Fat';




l = 0;
name = [];
label  = [];
for str = 1:5
    for n = 1:3
        for s = 1:3600
            l = l+1;
            name{l} = char(Str{str}+num2str(n)+'_'+num2str(s)+'.txt');
            label{l} =  LABEL{str};
        end

    end
end
T = table(name',label');
writetable(T,'Train.csv','Delimiter',',','QuoteStrings',true);
type('Train.csv');


l = 0;
name = [];
label  = [];
for str = 1:5
    for n = 4
        for s = 1:3600
            l = l+1;
            name{l} = char(Str{str}+num2str(n)+'_'+num2str(s)+'.txt');
            label{l} =  LABEL{str};
        end     
    end
end
T = table(name',label');
writetable(T,'Evaluate.csv','Delimiter',',','QuoteStrings',true);
type('Evaluate.csv');


l = 0;
name = [];
label  = [];
for str = 1:5
    for n = 5
        for s = 1:3600
            l = l+1;
            name{l} = char(Str{str}+num2str(n)+'_'+num2str(s)+'.txt');
            label{l} =  LABEL{str};
        end
    end
end
T = table(name',label');
writetable(T,'Test.csv','Delimiter',',','QuoteStrings',true);
type('Test.csv');
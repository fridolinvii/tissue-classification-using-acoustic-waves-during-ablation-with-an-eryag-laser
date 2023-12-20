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
for str = 1:5 %[3 1 5 2 4] %[4 2 5 1 3]
    for n = 1%:3
        for s = 1000%:3600
            l = l+1;
            name{l} = char(Str{str}+num2str(n)+'_'+num2str(s)+'.txt');
            label{l} =  LABEL{str};
        end

    end
end
T = table(name',label');
writetable(T,'Paper.csv','Delimiter',',','QuoteStrings',true);
type('Paper.csv');

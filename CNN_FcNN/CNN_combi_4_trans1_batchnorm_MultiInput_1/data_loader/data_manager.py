import os
import torch as th
import numpy as np
import matplotlib.pyplot as plt
import random
import torch.nn.functional as F
from utils import utils

from torch.utils import data
import numpy as np
import random

import scipy.io as sio

#####################################################################################################
#####################################################################################################

class DataManager(data.Dataset):
    def __init__(self, args, params, data,train=True):


        self._data = data
        self._train = train
        self.params = params
        self.args = args


        print("Number of spectra in the training set", len(self._data.data))

    def sequence_length(self):
        return len(self._data.data)

    def __len__(self):
        return len(self._data.data)


    def __getitem__(self, idx):


        # Here,is Output:
        #                 - Image
        #                 - Label
        data = {"time": 0, "fft":0}
    
        label = th.tensor(self.args.classes.index(self._data.label[idx]))
        data["time"] = th.tensor(self._data.data[idx],dtype=th.float)
        data["fft"] = th.tensor(self._data.data_fft[idx],dtype=th.float) #.unsqueeze(0);
        hole = 0 #th.tensor(self._data.hole[idx],dtype=th.long)
        day = 0 #th.tensor(self._data.day[idx],dtype=th.long)


#        print(data.shape)

        # cut to desired frequency range
        data["fft"], _ = utils.fittToFrequency(self._data.freq, data["fft"], self.args.frequencyrange)

        # data = data.squeeze(0)

        return data, label, hole, day




## Preaload all the data
class LoadData:
    def __init__(self, Speziment):


        Tissue = ('HB','SB','Fat','Skin','Muscle') #["Fat","HB","Muscle","SB","Skin"]
        Label  = ('HardBone','SoftBone','Fat','Skin','Muscle') #["Fat","HardBone","Muscle","SoftBone","Skin"]


        self.data = []
        self.data_fft = []
        self.label = []
        self.hole = []
        self.day = []

        path = "."

        Hole = 0;
        
        T = 2
        # Differntiate different type of tissues

        
        for s in Speziment:
            
            for t in range(len(Tissue)):
                           
                M = sio.loadmat(path+"/data/matrix_pre_3000/trans1/matrix_time/"+Tissue[t]+"/"+Tissue[t]+s+".mat")
                M_fft = sio.loadmat(path+"/data/matrix_pre_3000/trans1/matrix_fft/"+Tissue[t]+"/"+Tissue[t]+s+".mat")
#               
                N = M['signal'].shape[1]
                self.N = N

                print([s,t,M['signal'].shape[1]])

                Hole = -1;
                numberOfWaves = 1
                for n in range(5): # 5 holes
                    for k in range(100-numberOfWaves+1): # 100 shots
                        kk = n*100+k
                        MM = M['signal'][np.newaxis,:,kk]
                        MM_fft = M_fft['signal'][np.newaxis,:,kk]
                        for kn in range(numberOfWaves-1):
                            MM = np.concatenate((MM,M['signal'][np.newaxis,:,kn+kk+1]),0)
                            MM_fft = np.concatenate((MM_fft,M_fft['signal'][np.newaxis,:,kn+kk+1]),0)

                        self.label.append(Label[t])
                        self.data.append(MM)
                        self.data_fft.append(MM_fft)


                    
        M = sio.loadmat(path+"/data/matrix_pre_3000/trans1/matrix_fft/"+Tissue[t]+"/freq.mat")
        self.freq = M['f'][0,:]
        # self.NumHoles = 2
        self.NumHoles = 0 # Hole+1
        self.NumDays = 0 #day+1
#        print([Hole,day])





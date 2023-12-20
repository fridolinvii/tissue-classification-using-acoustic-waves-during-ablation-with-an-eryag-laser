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

        label = th.tensor(self.args.classes.index(self._data.label[idx]))
        data = th.tensor(self._data.data[idx],dtype=th.float) #.unsqueeze(0);
        hole = 0 #th.tensor(self._data.hole[idx],dtype=th.long)
        day = 0 #th.tensor(self._data.day[idx],dtype=th.long)


#        print(data.shape)

        # cut to desired frequency range
        # data, _ = utils.fittToFrequency(self._data.freq, data, self.args.frequencyrange)

        # data = data.squeeze(0)

        return data, label, hole, day




## Preaload all the data
class LoadData:
    def __init__(self, Speziment):


        Tissue = ('HB','SB','Fat','Skin','Muscle') #["Fat","HB","Muscle","SB","Skin"]
        Label  = ('HardBone','SoftBone','Fat','Skin','Muscle') #["Fat","HardBone","Muscle","SoftBone","Skin"]


        self.data = []
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
                        for kn in range(numberOfWaves-1):
                            MM = np.concatenate((MM,M['signal'][np.newaxis,:,kn+kk+1]),0)

                        self.label.append(Label[t])
                        self.data.append(MM)


#                for i in range(N):
#                    
#                    ## Differntiate different type of Speziment Day
#                    if (int(s)<6)  and (int(s)>0):
#                        day = 0
#                    else :
#                        day = 1
#                    self.day.append(day)
                    

                    ## differentiate which holes were cut first
#                    if 0 == i % 100:
#                        Hole += 1

#                    self.label.append(Label[t])
#                    self.data.append(M['signal'][:,i])
#                    if t == T:
#                        self.hole.append(Hole)
#                    else :
#                        # self.hole.append(-1)
#                        self.hole.append(Hole)

                    
        # M = sio.loadmat(path+"/data/matrix_pre_3000/trans1/matrix_fft/"+Tissue[t]+"/freq.mat")
        # self.freq = M['f'][0,:]
        # self.NumHoles = 2
        self.NumHoles = 0 # Hole+1
        self.NumDays = 0 #day+1
#        print([Hole,day])





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

import sklearn.decomposition as sd
import scipy.io as sio

#####################################################################################################
#####################################################################################################

class DataManager(data.Dataset):
    def __init__(self, args, params, data, train=True):


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
        #data, _ = utils.fittToFrequency(self._data.freq, data, self.args.frequencyrange)
        #data = data-self.pca.mu[None,:]
        #data = th.tensor(self.pca.pca.transform(data),dtype=th.float)

        # data = data.squeeze(0)

        return data, label, hole, day


#############################################################################################


class createPCADecomposition_precomputed:
    def __init__(self, args, data1,data2):
        data = []
        for i in range(len(data1.data)):
            if i % 5000 == 0:
                print("Load data1 for PCA: ", i, "/", len(data1.data))
            _data = th.tensor(data1.data[i],dtype=th.float)
            _data, _ = utils.fittToFrequency(data1.freq, _data, args.frequencyrange)
            if i == 0:
                data = _data[None,1,:]
                # data = _data
            else:
                data = th.cat((data,_data[None,1,:]),0)
                # data = th.cat((data,_data),0)
        for i in range(len(data2.data)):
            if i % 5000 == 0:
                print("Load data2 for PCA: ", i, "/", len(data2.data))
            _data = th.tensor(data2.data[i],dtype=th.float)
            _data, _ = utils.fittToFrequency(data2.freq, _data, args.frequencyrange)
            data = th.cat((data,_data[None,1,:]),0)
            # data = th.cat((data,_data),0)

        self.mu = th.mean(data,dim=0)
        self.data = data-self.mu[None,:]


class createPCADecomposition:
    def __init__(self, in_data, n_components=3):
        self.pca = sd.PCA(n_components=n_components)
        self.pca.fit(in_data.data)
        self.mu = in_data.mu


class createPCADecomposition_apply:
    def __init__(self, args, data, pca):
        self.data = []
        for i in range(len(data.data)):
            if i % 5000 == 0:
                print("Load data for PCA: ", i, "/", len(data.data))
            _data, freq = utils.fittToFrequency(data.freq, data.data[i], args.frequencyrange)
            _data = _data-pca.mu[None,:].numpy()
            self.data.append(th.tensor(pca.pca.transform(_data),dtype=th.float))
        
        self.freq = freq
        self.label = data.label

        

#############################################################################################

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
                numberOfWaves = 10
                for n in range(5): # 5 holes
                    for k in range(100-numberOfWaves): # 100 shots
                        kk = n*100+k
                        MM1 = M['signal'][np.newaxis,:,kk]
                        MM1 = np.concatenate((MM1,M['signal'][np.newaxis,:,kk+1]),0)
                        MM = np.mean(MM1,0)
                        MM = np.expand_dims(MM, axis=0)


                        for kn in range(numberOfWaves-1):
                            MM1 = M['signal'][np.newaxis,:,kk+kn+1]
                            MM1 = np.concatenate((MM1,M['signal'][np.newaxis,:,kk+kn+2]),0)
                            MM1 = np.mean(MM1,0)
                            MM1 = np.expand_dims(MM1, axis=0)
                            MM = np.concatenate((MM,MM1),0)

     
                        MM, freq = utils.convertToFrequency(MM)
                                              
                        self.label.append(Label[t])
                        self.data.append(MM)
                        

                    
        #M = sio.loadmat(path+"/data/matrix_pre_3000/trans1/matrix_fft/"+Tissue[t]+"/freq.mat")
        self.freq = freq# M['f'][0,:]
        # self.NumHoles = 2
        self.NumHoles = 0 # Hole+1
        self.NumDays = 0 #day+1
#        print([Hole,day])





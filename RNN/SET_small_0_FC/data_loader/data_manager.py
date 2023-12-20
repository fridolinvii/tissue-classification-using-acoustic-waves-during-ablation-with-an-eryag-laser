__author__ = "Carlo Seppi"
__copyright__ = "Copyright (C) 2019 Center for medical Image Analysis and Navigation"
__email__ = "carlo.seppi@unibas.ch"

import os
import torch as th
import pandas as pd
import numpy as np
import matplotlib as plt
import scipy.io as sio


from utils import utils

from torch.utils import data

import scipy.io as sio


#####################################################################################################
####################   Er:YAG #######################################################################
#####################################################################################################




class DataManager_Time(data.Dataset):
    def __init__(self, data, SampleRate, FrequencyRange, args=None):
        self._SampleRate = SampleRate
        self._FrequencyRange = FrequencyRange
        self._data = data
        _, _, self.input_size = self.preprocess(0)
        print("Number of spectra in the training set", len(self._data.data))

    def sequence_length(self):
        return len(self._data.data)
    
    def preprocess(self, idx):
        classes = ('HardBone','SoftBone','Fat','Skin','Muscle')
    
        label = self._data.label[idx]
        image = self._data.data[idx]
        
        sample = {'image': image, 'label': label}
        image = th.from_numpy(image).float()        
        label = th.tensor(classes.index(sample['label']))
        
        image = image.numpy()*np.hamming(image.shape[-1])
        image = image/max(abs(image)+1e-10)
        image = utils.frequencyFilter(self._SampleRate,self._FrequencyRange,image)
        image = th.from_numpy(image).float()
        label = th.tensor(classes.index(sample['label']))

        input_size = image.shape[-1]

        return image, label, input_size
    
    def __len__(self):
        return len(self._data.data)

    def __getitem__(self, idx):
    
        image, label, _ = self.preprocess(idx)


        return image, label








        
        
        
        
class DataManager_Frequency(data.Dataset):
    def __init__(self, data, SampleRate, FrequencyRange, args=None):

        self._SampleRate = SampleRate
        self._FrequencyRange = FrequencyRange
        self._data = data
        _, _, self.input_size = self.preprocess(0)
        print("Number of spectra in the training set", len(self._data.data))

    def sequence_length(self):
        return len(self._data.data)
    
    def preprocess(self, idx):
        classes = ('HardBone','SoftBone','Fat','Skin','Muscle')
    
        label = self._data.label[idx]
        image = self._data.data[idx]

        sample = {'image': image, 'label': label}

        image = th.from_numpy(image).float()        
        label = th.tensor(classes.index(sample['label']))
        
        image = image.numpy()*np.hamming(len(image))  #normalize time instead of frequency
        image = image/max(abs(image)+1e-10) 
        image, Freq = utils.timeToFrequency(self._SampleRate,self._FrequencyRange,image)
        image = th.from_numpy(image)
        image = image.float()

        input_size = image.shape[-1]

        return image, label, input_size
    
    def __len__(self):
        return len(self._data.data)

    def __getitem__(self, idx):
    

        image, label, _ = self.preprocess(idx)

        return image, label



class LoadData:
    def __init__(self, Speziment, path):


        Tissue = ('HB','SB','Fat','Skin','Muscle') #["Fat","HB","Muscle","SB","Skin"]
        Label  = ('HardBone','SoftBone','Fat','Skin','Muscle') #["Fat","HardBone","Muscle","SoftBone","Skin"]


        self.data = []
        self.label = []
        self.hole = []
        self.day = []

        


        for s in Speziment:

            for t in range(len(Tissue)):

                M = sio.loadmat("../../"+path+"/"+Tissue[t]+"/"+Tissue[t]+s+".mat")
#
                N = M['A'].shape[1]

                self.N = N
                print([s,t,M['A'].shape[1]])

                numberOfWaves = 1 #5 #15
                for n in range(5): # 5 holes
                    for k in range(100-numberOfWaves+1): # 100 shots
                        kk = n*100+k
                        MM = M['A'][np.newaxis,:,kk]
                        for kn in range(numberOfWaves-1):
                            MM = np.concatenate((MM,M['A'][np.newaxis,:,kn+kk+1]),0)

                        self.label.append(Label[t])
                        self.data.append(MM)


     
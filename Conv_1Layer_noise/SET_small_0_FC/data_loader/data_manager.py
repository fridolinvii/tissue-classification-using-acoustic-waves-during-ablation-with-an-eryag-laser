__author__ = "Carlo Seppi"
__copyright__ = "Copyright (C) 2019 Center for medical Image Analysis and Navigation"
__email__ = "carlo.seppi@unibas.ch"

import os
import torch as th
import pandas as pd
import numpy as np
import matplotlib as plt


from utils import utils

from torch.utils import data



#####################################################################################################
####################   Er:YAG #######################################################################
#####################################################################################################







class DataManager_Time_noise(data.Dataset):
    def __init__(self, path, csvpath, TimeRange, SampleRate,FrequencyRange,noise=True, num_class=5, args=None):

        self._training_set = []
        self._path = path
        self._csvpath = csvpath
        self._args = args
        self._num_class = num_class
        self._labels_frame = pd.read_csv(self._csvpath)
        self._SampleRate = SampleRate
        self._FrequencyRange = FrequencyRange
        self._TimeRange = TimeRange
        self._noise = noise
        print("Number of spectra in the training set", len(self._labels_frame))

    def sequence_length(self):
        return len(self._labels_frame)

    def __len__(self):
        return len(self._labels_frame)

    def __getitem__(self, idx):
    
        classes = ('HardBone','SoftBone','Fat','Skin','Muscle')
    
    
        img_name = os.path.join(self._path,
                                self._labels_frame.iloc[idx, 0])
        image = np.loadtxt(img_name)
        label = self._labels_frame.iloc[idx, 1]
        
        
        sample = {'image': image, 'label': label}

        if self._noise is True:
            shift = th.randint(-50,51,(1,))
           # p = th.rand(1)*0.15
            self._noise = [shift[0],0] #p[0]]
        elif self._noise is False:
            self._noise = [0,0] 

        # time window plus shift
        image = image[(self._TimeRange[0]+self._noise[0]):(self._TimeRange[1]+self._noise[0])]
        image = th.from_numpy(image).float()

        # add p gauss noise N(0,1)
        image = image*(1+self._noise[1]*th.log(1+abs(image)))


        label = th.tensor(classes.index(sample['label']))
        

        image = image.numpy()*np.hamming(len(image))
        image = image/max(abs(image))
        image = utils.frequencyFilter(self._SampleRate,self._FrequencyRange,image)
        image = th.from_numpy(image).float()
        label = th.tensor(classes.index(sample['label']))
        
        
        
        return image, label


##############################################################################################################
##############################################################################################################
##############################################################################################################


class DataManager_Frequency_noise(data.Dataset):
    def __init__(self, path, csvpath, TimeRange, SampleRate,FrequencyRange,noise=True, num_class = 5, args=None):

        self._training_set = []
        self._path = path
        self._csvpath = csvpath
        self._args = args
        self._num_class = num_class
        self._labels_frame = pd.read_csv(self._csvpath)
        self._SampleRate = SampleRate
        self._FrequencyRange = FrequencyRange
        self._TimeRange = TimeRange
        self._noise = noise
        print("Number of spectra in the training set", len(self._labels_frame))

    def sequence_length(self):
        return len(self._labels_frame)

    def __len__(self):
        return len(self._labels_frame)

    def __getitem__(self, idx):
    
        classes = ('HardBone','SoftBone','Fat','Skin','Muscle')
    
    
        img_name = os.path.join(self._path,
                                self._labels_frame.iloc[idx, 0])
        image = np.loadtxt(img_name)
        label = self._labels_frame.iloc[idx, 1]
        
        
        sample = {'image': image, 'label': label}


        if self._noise is True:
            shift = th.randint(-50,51,(1,))
           # p = th.rand(1)*0.15
            self._noise = [shift[0],0] #p[0]]
        elif self._noise is False:
            self._noise = [0,0] 

        # time window plus shift
        image = image[(self._TimeRange[0]+self._noise[0]):(self._TimeRange[1]+self._noise[0])]
        image = th.from_numpy(image).float()

        # add p gauss noise N(0,1)
        image = image*(1+self._noise[1]*th.log(1+abs(image)))


        label = th.tensor(classes.index(sample['label']))
#        image = image.numpy()*np.hamming(len(image))
        image = image.numpy()*np.hamming(len(image))  #normalize time instead of frequency
        image = image/max(abs(image))
#        image = image.numpy()*np.hamming(len(image)) 
        image, Freq = utils.timeToFrequency(self._SampleRate,self._FrequencyRange,image)
#        image, Freq = utils.timeToFrequency(self._SampleRate,self._FrequencyRange,image.numpy())
#        image = image/max(image) #*self._mult;
        image = th.from_numpy(image)
        image = image.float()

        return image, label



class DataManager_Paper(data.Dataset):
    def __init__(self, path, csvpath, TimeRange, SampleRate,FrequencyRange,num_class = 5, args=None):

        self._training_set = []
        self._path = path
        self._csvpath = csvpath
        self._args = args
        self._num_class = num_class
 
        self._labels_frame = pd.read_csv(self._csvpath)
        
        self._SampleRate = SampleRate
        self._FrequencyRange = FrequencyRange
        self._TimeRange = TimeRange
        
        print("Number of spectra in the training set", len(self._labels_frame))

    def sequence_length(self):
        return len(self._labels_frame)

    def __len__(self):
        return len(self._labels_frame)

    def __getitem__(self, idx):
    



        classes = ('HardBone','SoftBone','Fat','Skin','Muscle')
    
    
        img_name = os.path.join(self._path,
                                self._labels_frame.iloc[idx, 0])
        image = np.loadtxt(img_name)
        label = self._labels_frame.iloc[idx, 1]
        
        
        sample = {'image': image, 'label': label}
        label = th.tensor(classes.index(sample['label']))
        
        image = image[self._TimeRange[0]:self._TimeRange[1]]
        image = th.from_numpy(image).float()        
        
        image_true = image
        
        

        image = image.numpy()*np.hamming(len(image))
        image = image/max(abs(image))
        
        image_time = utils.frequencyFilter(self._SampleRate,self._FrequencyRange,image)
        image_freq, Freq = utils.timeToFrequency(self._SampleRate,self._FrequencyRange,image)


#        image_true = th.from_numpy(image_true).float()
        image_freq = th.from_numpy(image_freq).float()
        image_time = th.from_numpy(image_time).float()

        
        
        
        Frame = 1 / self._SampleRate
        L = len(image)

        Time = Frame*L;
        time = np.linspace(0.0, Time , L)
        time = th.from_numpy(time).float()
        
        
        return image_true, image_time, image_freq, Freq, time , label








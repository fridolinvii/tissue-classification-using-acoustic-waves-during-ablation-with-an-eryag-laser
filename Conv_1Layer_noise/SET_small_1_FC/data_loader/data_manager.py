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




class DataManager_Time(data.Dataset):
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
        image = image[self._TimeRange[0]:self._TimeRange[1]]
        image = th.from_numpy(image).float()        
        label = th.tensor(classes.index(sample['label']))
        

        image = image.numpy()*np.hamming(len(image))
        image = image/max(abs(image))
        image = utils.frequencyFilter(self._SampleRate,self._FrequencyRange,image)
#        print(max(image))
#        image = image[0:self._TimeRange]
#        image = abs(image)/max(abs(image));
#        image = image/max(abs(image));
        image = th.from_numpy(image).float()
        label = th.tensor(classes.index(sample['label']))
        
        
        
        return image, label


        
        
        
        
class DataManager_Frequency(data.Dataset):
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
        image = image[self._TimeRange[0]:self._TimeRange[1]]
        image = th.from_numpy(image).float()        
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
        
        
        

class DataManager_Frequency_log(data.Dataset):
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
        image = image[self._TimeRange[0]:self._TimeRange[1]]
        image = th.from_numpy(image).float()        
        label = th.tensor(classes.index(sample['label']))
        
#        image = image.numpy()*np.hamming(len(image))
        image = image.numpy()*np.hamming(len(image))  #normalize time instead of frequency
        image = image/max(abs(image))
#        image = image.numpy()*np.hamming(len(image)) 
        image, Freq = utils.timeToFrequency(self._SampleRate,self._FrequencyRange,image)
#        image, Freq = utils.timeToFrequency(self._SampleRate,self._FrequencyRange,image.numpy())
#        image = image/max(image) #*self._mult;
        image = np.log(image+1)
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







#####################################################################################################
####################   Nd:YAG #######################################################################
#####################################################################################################

class DataManager_ndyag_Frequency_2(data.Dataset):
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
    
    
        img_name_1 = os.path.join(self._path, self._labels_frame.iloc[idx, 1])
        img_name_2 = os.path.join(self._path, self._labels_frame.iloc[idx, 2])
        image_1 = np.loadtxt(img_name_1)
        image_2 = np.loadtxt(img_name_2)
        label = self._labels_frame.iloc[idx, 0]
        
        
        sample = {'label': label}
        label = th.tensor(classes.index(sample['label']))

        image_1 = image_1[self._TimeRange[0]:self._TimeRange[1]]
        image_1 = th.from_numpy(image_1).float()        
        image_2 = image_2[self._TimeRange[0]:self._TimeRange[1]]
        image_2 = th.from_numpy(image_2).float()        

        

        image_1 = image_1.numpy()*np.hamming(len(image_1))  
        image_2 = image_2.numpy()*np.hamming(len(image_2))  #normalize time instead of frequency
        image_1 = image_1/max(abs(image_1))
        image_2 = image_2/max(abs(image_2))
        image_1, Freq = utils.timeToFrequency(self._SampleRate,self._FrequencyRange,image_1)
        image_2, Freq = utils.timeToFrequency(self._SampleRate,self._FrequencyRange,image_2)
        image = (image_1+image_2)/2
        image = th.from_numpy(image)
        image = image.float()

            
        
        return image, label
        


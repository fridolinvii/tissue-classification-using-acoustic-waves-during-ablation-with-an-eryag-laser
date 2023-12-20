_author__ = "Carlo Seppi, Eva Schnider"
__copyright__ = "Copyright (C) 2020 Center for medical Image Analysis and Navigation"
__email__ = "carlo.seppi@unibas.ch"

import numpy as np

import torch.nn as nn
import torch.nn.functional as F
# import torch.optim as optim
import torch as th
#from nnAudio import Spectrogram

import matplotlib.pyplot as plt



###########################################################################################
###########################################################################################
###########################################################################################


class ParametricFCModel(nn.Module):

    def __init__(self, image_all, args, params):
        super(ParametricFCModel, self).__init__()

        image = image_all["time"]
        image_fft = image_all["fft"]


        ##################################################################################
        # Find number of input 
        input_size = image.shape[-1]
        in_channels = image.shape[0]

        self.acti_func = F.relu
        self.FC_Layer = nn.ModuleDict();

        for i in range(in_channels):
            self.FC_Layer["BN0_"+str(i)] =   nn.BatchNorm1d(input_size)
            self.FC_Layer["BNfft0_"+str(i)] =   nn.BatchNorm1d(image_fft.shape[-1])
#        self.FC_Layer["BN0"] =   nn.BatchNorm1d(in_channels)
        ## CNN Layers ##
        count = 0
        for i in range(params['n_cnn_layers']):

            padding = 0 #int(params['cnn_kernel_size']/2)

            # check if inputSize is bigger than 1
            input_size_ = np.floor(((input_size + 2*padding- 1 * (params['cnn_kernel_size'] - 1) - 1) / 1) + 1)
            input_size_ = np.floor(((input_size_ - 1 * (params['max_pooling_kernel_size'] - 1) - 1) / params['max_pooling_kernel_size']) + 1)

            if input_size_ > 10:
                input_size = input_size_
                if i == 0:
                    layerCNN = nn.Conv1d(in_channels=in_channels, out_channels=(i+1)*params['cnn_output_channels'], kernel_size=params['cnn_kernel_size'],padding=padding)
                else:
                    layerCNN = nn.Conv1d(in_channels=i*params['cnn_output_channels'], out_channels=(i+1)*params['cnn_output_channels'], kernel_size=params['cnn_kernel_size'],padding=padding)
                self.FC_Layer["CNN"+str(count)]=layerCNN
                self.FC_Layer["MaxPool"+str(count)]=nn.MaxPool1d(kernel_size=params['max_pooling_kernel_size'])
                count += 1
            else:
                break

                   
        
        input_size = count*int(params['cnn_output_channels']*input_size)
        self.FC_Layer["BN1"] =   nn.BatchNorm1d(input_size)
        input_size += image_fft.shape[-1]*image_fft.shape[0]
        self.n_cnn_layers = count
        

        ## FC Layers ##
        count = 0
        for i in range(params['n_fc_layers']+1):
            if i == 0:
                layerFC = nn.Linear(input_size, params["neurons"])
            elif i == params['n_fc_layers']:
                layerFC = nn.Linear(params["neurons"], len(args.classes))
                
            else :
                layerFC = nn.Linear(params["neurons"], params["neurons"])
            self.FC_Layer["Dropout"+str(count)] = nn.Dropout(p=params['dropout'])
            self.FC_Layer[str(count)]=layerFC

            count += 1
        
        self.error = False

        self.n_fc_layers = count
        self.tsne_layer = params['tsne_layer']
    ##################################################################################################
    ##################################################################################################
    ## Define the Forward Model ##
    def forward(self, xin,xin_fft):


        for i in range(xin.shape[1]):
            xB = self.FC_Layer["BN0_"+str(i)](xin[:,i,:]).unsqueeze(1)
            if i == 0:
                x = xB
            else:
                x = th.cat((x,xB),1)

        for i in range(xin_fft.shape[1]):
            xB = self.FC_Layer["BNfft0_"+str(i)](xin_fft[:,i,:]).unsqueeze(1)
            if i == 0:
                x_fft = xB
            else:
                x_fft = th.cat((x_fft,xB),1)


#        x = self.FC_Layer["BN0"](xin) #.squeeze(1)).unsqueeze(1)

        ## CNN Layers ##
        for i in range(self.n_cnn_layers):
            x = self.acti_func(self.FC_Layer["CNN"+str(i)](x))
            x = self.FC_Layer["MaxPool"+str(i)](x)
            
        x = th.flatten(x, 1)
        x = self.FC_Layer["BN1"](x)
        x_fft = th.flatten(x_fft, 1)
        x = th.cat((x,x_fft),1)
        
        ## FC Layers ##
        for i in range(self.n_fc_layers) :                 
            if i == self.n_fc_layers-1:
                x = self.FC_Layer["Dropout"+str(i)](x)
                x_out = self.FC_Layer[str(self.n_fc_layers-1)](x)
            else:
                x = self.FC_Layer["Dropout"+str(i)](x)
                x = self.FC_Layer[str(i)](x)
                x = self.acti_func(x)
            
            if i == self.tsne_layer:
                x_hole = 0
                x_day = 0
                tsne = x   


        return  x_out, x_hole, x_day, tsne




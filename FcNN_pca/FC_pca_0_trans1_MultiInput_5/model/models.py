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

    def __init__(self, image, args, params):
        super(ParametricFCModel, self).__init__()





        ##################################################################################
        # Find number of input 
        input_size = image.shape[-1]
        in_channels = image.shape[0]

        self.acti_func = F.relu
        self.FC_Layer = nn.ModuleDict();

#        for i in range(in_channels):
#            self.FC_Layer["BN0_"+str(i)] =   nn.BatchNorm1d(input_size)
#        self.FC_Layer["BN0"] =   nn.BatchNorm1d(in_channels)


        input_size =int(in_channels*input_size)
#        self.FC_Layer["BN1"] =   nn.BatchNorm1d(input_size)
        

        ## FC Layers ##
        count = 0
        for i in range(params['n_fc_layers']+1):
            if i == 0:
                layerFC = nn.Linear(input_size, params["neurons"])
            elif i == params['n_fc_layers']:
                layerFC = nn.Linear(params["neurons"], len(args.classes))
#                layerFC = nn.Linear(in_channels*params["neurons"], len(args.classes))
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
    def forward(self, xin):


#        for i in range(xin.shape[1]):
#            xB = self.FC_Layer["BN0_"+str(i)](xin[:,i,:]).unsqueeze(1)
#            if i == 0:
#                x = xB
#            else:
#                x = th.cat((x,xB),1)



            
        x = th.flatten(xin, 1)
#        x = self.FC_Layer["BN1"](x)
        ## FC Layers ##
        for i in range(self.n_fc_layers) :                 
            if i == self.n_fc_layers-1:
#                x = th.flatten(x, 1)
                x = self.FC_Layer["Dropout"+str(i)](x)
                x_out = self.FC_Layer[str(self.n_fc_layers-1)](x)
            else:
                x = self.FC_Layer["Dropout"+str(i)](x)
                x = self.FC_Layer[str(i)](x)
                x = self.acti_func(x)
            
        x_hole = 0
        x_day = 0
        tsne = x   


        return  x_out, x_hole, x_day, tsne




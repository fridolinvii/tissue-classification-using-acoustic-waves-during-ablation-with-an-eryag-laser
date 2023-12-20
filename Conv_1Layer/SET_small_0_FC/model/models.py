__author__ = "Carlo Seppi"
__copyright__ = "Copyright (C) 2019 Center for medical Image Analysis and Navigation"
__email__ = "carlo.seppi@unibas.ch"




import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch



#####################################################################################################
####################   Er:YAG #######################################################################
#####################################################################################################

class ConvClasi_frequency_length_1(nn.Module):
    def __init__(self, input_size, num_classes):
        super(ConvClasi_frequency_length_1, self).__init__()


        channels_in = 1
        channels_out = 6
        kernel_size = 2


        self.conv0 = nn.Conv1d(1, channels_out, kernel_size)
        self.fc1 = nn.Linear(3*(input_size-kernel_size), 1000)
        self.fc2 = nn.Linear(1000, 1000) #freq = [1e5, 1e6]
        self.fc3 = nn.Linear(1000, num_classes)

        
        self.cam_feature_maps = None
        self.outputs = None
        
      


    def forward(self, x):



        self.inputs = x


        x = F.max_pool1d(F.relu(self.conv0(x)), 2)
        self.grad_cam_feature_maps = x
        x = x.view(-1, self.num_flat_features(x))  

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        self.classifications = torch.argmax(x, dim=1)
        self.outputs = F.softmax(x,dim=1)

        return x



    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features





#####################################################################################3



class ConvClasi_time_length(nn.Module):
    def __init__(self, input_size, num_classes):
        super(ConvClasi_time_length, self).__init__()


        channels_in = 1
        channels_out = 6
        kernel_size = 200


        self.conv0 = nn.Conv1d(1, channels_out, kernel_size,stride=1)
        self.fc1 = nn.Linear(3*(input_size-kernel_size), 1000)        
        self.fc2 = nn.Linear(1000, 1000)        
        self.fc3 = nn.Linear(1000, num_classes)

        self.cam_feature_maps = None
        self.outputs = None
        
        
        

    def forward(self, x):



        self.inputs = x

        x = F.max_pool1d(F.relu(self.conv0(x)), 2)
        self.grad_cam_feature_maps = x

        x = x.view(-1, self.num_flat_features(x))  
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        self.classifications = torch.argmax(x, dim=1)
        self.outputs = F.softmax(x,dim=1)

        return x


    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    




















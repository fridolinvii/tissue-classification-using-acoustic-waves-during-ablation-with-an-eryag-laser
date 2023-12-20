__author__ = "Carlo Seppi"
__copyright__ = "Copyright (C) 2019 Center for medical Image Analysis and Navigation"
__email__ = "carlo.seppi@unibas.ch"

import torch as th
import numpy as np
import visdom as vis

from data_loader import data_manager as dm
from torch.utils import data
from model import models as m
from utils import utils

import torch.nn as nn
import torch.nn.functional as F


import matplotlib.pyplot as plt

Mean = np.zeros((5, 5))

number = input ("Enter number: ")


path = 'data/ErYAG/frequency_results_length_'+number+'/model_best.pt'
state = th.load(path, map_location="cpu")
args_load = state['args']
args_load.path = '/home/carlo/Documents/Deeplearning/Tissue_Differentiation_Microphone_25.07.2019/data/ErYAG/time/'
# testcsv_path
data_manager = dm.DataManager_Frequency(args_load.path,args_load.traincsv_path, args_load.timerange, args_load.samplerate, args_load.frequencyrange,  args=None) 

if (int(number) == 0):
    model = m.ConvClasi_frequency_length_0(data_manager.sequence_length(), num_classes=len(args_load.classes))
if (int(number) == 1):
    model = m.ConvClasi_frequency_length_1(data_manager.sequence_length(), num_classes=len(args_load.classes))
    alphaM = 6
if (int(number) == 2):
    model = m.ConvClasi_frequency_length_2(data_manager.sequence_length(), num_classes=len(args_load.classes))
if (int(number) == 3):
    model = m.ConvClasi_frequency_length_3(data_manager.sequence_length(), num_classes=len(args_load.classes))
if (int(number) == 4):
    model = m.ConvClasi_frequency_length_4(data_manager.sequence_length(), num_classes=len(args_load.classes))
if (int(number) == 5):
    model = m.ConvClasi_frequency_length_5(data_manager.sequence_length(), num_classes=len(args_load.classes))
if (int(number) == 6):
    model = m.ConvClasi_frequency_length_6(data_manager.sequence_length(), num_classes=len(args_load.classes))

model.load_state_dict(state['network'])

device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
print(device)
model.to(device)
model = model.eval()
print(model)
print(state['epoch'])

params = {'batch_size': 1, #args_load.batch_size,
          'shuffle': False,
          'num_workers': 10, # args_load.nb_workers,
          'pin_memory': True}
training_generator = data.DataLoader(data_manager, **params)

#	print(model)
total = 0
correct = 0

#    class_correct = list(0. for i in range(len(args_load.classes)))
#    class_total = list(0. for i in range(len(args_load.classes)))
class_predict = np.zeros((len(args_load.classes), len(args_load.classes)))
class_total = np.zeros((len(args_load.classes), len(args_load.classes)))


criterion =nn.CrossEntropyLoss()
optimizer = th.optim.Adam(model.parameters(), lr=args_load.lr, amsgrad=args_load.amsgrad)


grad_cam = []


alpha0 = 0;
alpha1 = 0;
alpha2 = 0;
alpha3 = 0;
alpha4 = 0;

for spec, target_label in training_generator:

        inputs = spec.unsqueeze(1) #.to(device)
        inputs *= alphaM
        inputs = inputs.to(device)
        labels = target_label.to(device)
       
        output = model(inputs)





        ## Grad Cam ##

        optimizer.zero_grad()
        model.grad_cam_feature_maps.retain_grad() # zero_grad()
        
        
        loss_value = criterion(model.outputs, model.classifications)
 

        loss_value.backward()
        

        
        feature_map_gradients = model.grad_cam_feature_maps.grad
#        print(feature_map_gradients.size())

        alpha = 0
        for ii in range(6):
            g = feature_map_gradients[0,ii,:]
            alpha += g
        alpha = F.relu(alpha)
        
#        alpha  = F.relu(feature_map_gradients[0,0,:])
#        alpha += F.relu(feature_map_gradients[0,1,:])
#        alpha += F.relu(feature_map_gradients[0,2,:])
#        alpha += F.relu(feature_map_gradients[0,3,:])
#        alpha += F.relu(feature_map_gradients[0,4,:])
#        alpha += F.relu(feature_map_gradients[0,5,:])

#        alpha =  (feature_map_gradients[0,0,:])
#        alpha += (feature_map_gradients[0,1,:])
#        alpha += (feature_map_gradients[0,2,:])
#        alpha += (feature_map_gradients[0,3,:])
#        alpha += (feature_map_gradients[0,4,:])
#        alpha += (feature_map_gradients[0,5,:])

#        alpha = alpha-min(alpha)
#        alpha = alpha/max(alpha)
        alpha = alpha.cpu().numpy()


#        plt.plot(alpha)
#        plt.show()



#        print(alpha)
#        print(alpha/max(alpha))


        if (model.classifications == 0):
            alpha0 += alpha
        if (model.classifications == 1):
            alpha1 += alpha
        if (model.classifications == 2):
            alpha2 += alpha
        if (model.classifications == 3):
            alpha3 += alpha
        if (model.classifications == 4):
            alpha4 += alpha






        _, predicted = th.max(output.data, 1)

        #            c = (predicted == labels).squeeze()
        for i in range(params['batch_size']):#range(args_load.batch_size):
            class_predict[labels[i], predicted[i]] += 1
            class_total[labels[i], labels[i]] += 1

        #                label = labels[i]
        #                class_correct[label] += c[i].item()
        #                class_total[label] += 1

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('************* TEST DATA *******************')
print('Accuracy of the network on the test images: %.5f %%' % (
        100 * correct / total))
#    file.write('************* TEST1 DATA ******************* \n')
#    file.write('Accuracy of the network on the test images: %.5f %% \n' % (
#        100 * correct / total))

np.set_printoptions(suppress=True)
print(class_predict)



#plt.plot(alpha0/max(alpha0))
#plt.show()
#plt.plot(alpha1/max(alpha1))
#plt.show()
#plt.plot(alpha2/max(alpha2))
#plt.show()
#plt.plot(alpha3/max(alpha3))
#plt.show()
#plt.plot(alpha4/max(alpha4))
#plt.show()


#alpha = alpha0+alpha1+alpha2+alpha3+alpha4
#plt.plot(alpha/max(alpha))
#plt.show()




np.savetxt('paper/train_normalized/range0_eryag_frequency_length_'+number+'.csv', alpha0)
np.savetxt('paper/train_normalized/range1_eryag_frequency_length_'+number+'.csv', alpha1)
np.savetxt('paper/train_normalized/range2_eryag_frequency_length_'+number+'.csv', alpha2)
np.savetxt('paper/train_normalized/range3_eryag_frequency_length_'+number+'.csv', alpha3)
np.savetxt('paper/train_normalized/range4_eryag_frequency_length_'+number+'.csv', alpha4)





###########################################################################












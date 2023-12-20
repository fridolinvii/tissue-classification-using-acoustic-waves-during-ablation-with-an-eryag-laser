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

# evaluatecsv_path
# eval_path

number = input ("Enter number: ")


path = 'data/ErYAG/frequency_results_length_'+number+'/model_best.pt'


number = int(number)

state = th.load(path, map_location="cpu")
#state = th.load(path)
args_load = state['args']
args_load.path = '/home/carlo/Documents/Deeplearning/Tissue_Differentiation_Microphone_25.07.2019/data/ErYAG/time/'
data_manager = dm.DataManager_Frequency(args_load.path,args_load.evaluatecsv_path, args_load.timerange, args_load.samplerate, args_load.frequencyrange,  args=None) 
#data_manager = dm.DataManager_Frequency(args_load.path,args_load.traincsv_path, args_load.timerange, args_load.samplerate, args_load.frequencyrange,  args=None) 
#if (number == 0):
#    model = m.ConvClasi_frequency_length_0(data_manager.sequence_length(), num_classes=len(args_load.classes))
if (number == 1):
   alpha = 6
#    model = m.ConvClasi_frequency_length_1(data_manager.sequence_length(), num_classes=len(args_load.classes))
if (number == 2):
   alpha = 6
#    model = m.ConvClasi_frequency_length_2(data_manager.sequence_length(), num_classes=len(args_load.classes))
if (number == 3):
   alpha = 6
#    model = m.ConvClasi_frequency_length_3(data_manager.sequence_length(), num_classes=len(args_load.classes))
if (number == 4):
   alpha = 6
#    model = m.ConvClasi_frequency_length_4(data_manager.sequence_length(), num_classes=len(args_load.classes))
if (number == 5):
   alpha = 6
#    model = m.ConvClasi_frequency_length_5(data_manager.sequence_length(), num_classes=len(args_load.classes))
if (number == 6):
   alpha = 6
#    model = m.ConvClasi_frequency_length_6(data_manager.sequence_length(), num_classes=len(args_load.classes))
if (number == 7):
   alpha = 6
#    model = m.ConvClasi_frequency_length_7(data_manager.sequence_length(), num_classes=len(args_load.classes))


model = m.ConvClasi_frequency_length_1(data_manager.sequence_length(), num_classes=len(args_load.classes))



##### show number of parameters #####
pp=0
for p in list(model.parameters()):
    nn_=1
    for s in list(p.size()):
        nn_ = nn_*s
    pp += nn_

print(pp)
########################################3

model.load_state_dict(state['network'])

device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
print(device)
model.to(device)
model = model.eval()
print(state['epoch'])

params = {'batch_size': args_load.batch_size,
          'shuffle': False,
          'num_workers': 10, #args_load.nb_workers,
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


with th.no_grad():
     model = model.eval()
     for spec, target_label in training_generator:

        inputs = spec.unsqueeze(1).to(device)
        inputs *= alpha
        labels = target_label.to(device)
       
        output = model(inputs)

        _, predicted = th.max(output.data, 1)

        #            c = (predicted == labels).squeeze()
        for i in range(spec.shape[0]): #range(params['batch_size']):#range(args_load.batch_size):
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




###########################################################################











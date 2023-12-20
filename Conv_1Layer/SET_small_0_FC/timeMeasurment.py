import os
import torch as th
import pandas as pd
import numpy as np
import matplotlib as plt

from utils import utils
from torch.utils import data
from model import models as m

import time
import random

mod = input("0 is Time, 1 is Frequency: ")
cuda = input("0 cuda,  1 is cpu: ")
#number = input ("Enter number: ")



total_time = 0
for number in range(1,5):
    if "1" is mod:
        path = 'data/ErYAG/frequency_results_length_'+str(number)+'/model_best.pt'
        model = m.ConvClasi_frequency_length_1(0, num_classes=5)
        alpha = 6
    else:
        path = 'data/ErYAG/time_results_length_'+str(number)+'/model_best.pt'
        model = m.ConvClasi_time_length(0, num_classes=5)
        alpha = 1



    print(path)
#    print(model)

#    number = int(number)
    state = th.load(path, map_location="cpu")

    args = state['args']
    epoch_start = state['epoch']+1


    ## Load Model ##
    model.load_state_dict(state['network'])

    print("******************** CUDA ***********************")
#"    device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
    if cuda is "0":
        device="cuda:0"
    else :
        device="cpu"
    print(device)
    model.to(device)
    model = model.eval()
    print(state['epoch'])



    labels_frame = pd.read_csv(args.testcsv_path)

    classes = ('HardBone','SoftBone','Fat','Skin','Muscle')
    elapsed = 0
    LEN_DATA = 100 #len(labels_frame)
    correct = 0
    for idx in range(LEN_DATA):

        idx = random.choice(range(len(labels_frame)))


        img_name = os.path.join(args.path,
                                labels_frame.iloc[idx, 0])
        image = np.loadtxt(img_name)
        label = labels_frame.iloc[idx, 1]

        #########################################################################
        #########################################################################
        t = time.time()

        ## calculate model ##
        sample = {'image': image, 'label': label}
        image = image[args.timerange[0]:args.timerange[1]]
        image = th.from_numpy(image).float()
        label = th.tensor(classes.index(sample['label']))


        image = image.numpy()*np.hamming(len(image))
        image = image/max(abs(image))

        if "1" is mod:
            image, Freq = utils.timeToFrequency(args.samplerate, args.frequencyrange,image)
        else:
            image = utils.frequencyFilter(args.samplerate, args.frequencyrange,image)

        image = th.from_numpy(image).float()
        label = th.tensor(classes.index(sample['label']))


        with th.no_grad():
            inputs = image.unsqueeze(0).unsqueeze(0).to(device)
            inputs *= alpha
            labels = label.to(device)

            output = model(inputs)

            _, predicted = th.max(output.data, 1)
            correct += (predicted == labels).sum().item()


        elapsed += time.time() - t

    #########################################################################
    #########################################################################





    print("Time in s:")
    print(elapsed/LEN_DATA)
    print("Accuracy: ")
    print(correct/LEN_DATA)


    total_time += elapsed/LEN_DATA


print("\n*****************************************************")
print("----------------------------------------------------")
print("Mean Total Time:")
print(total_time/4)

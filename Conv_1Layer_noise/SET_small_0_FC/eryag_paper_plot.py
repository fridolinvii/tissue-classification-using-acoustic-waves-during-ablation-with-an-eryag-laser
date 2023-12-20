__author__ = "Carlo Seppi"
__copyright__ = "Copyright (C) 2019 Center for medical Image Analysis and Navigation"
__email__ = "carlo.seppi@unibas.ch"


import torch as th
import numpy as np
import visdom as vis


from data_loader import data_manager as dm
from torch.utils import data
from model import models as m




length = '4'
from parameter import parameter_eryag_time_length_4 as param



from utils import utils

import matplotlib.pyplot as plt
from tempfile import TemporaryFile


args = param.parser.parse_args()
path_csv = 'data/Paper.csv'
# Parameters
params = {'batch_size': 1, #args.batch_size,
            'shuffle': False,
            'num_workers': args.nb_workers,
            'pin_memory': True}





data_manager = dm.DataManager_Paper(args.path,path_csv, args.timerange, args.samplerate, args.frequencyrange,  args=None)
training_generator = data.DataLoader(data_manager, **params)





classes = ('HardBone','SoftBone','Fat','Skin','Muscle')

timerange = args.timerange[1]-args.timerange[0]
count = 0
for  image_true, image_time, image_freq, Freq , time, label in training_generator:
    
    Freq = Freq[0].numpy()
    time = time[0].numpy()

    if (count == 0):
        
        image_true_out = np.zeros((timerange,5))
        image_time_out = np.zeros((timerange,5))
        image_freq_out = np.zeros((len(Freq),5))
        label_array = np.zeros((5,1))
        
    label_array[count] = label.numpy().astype(int)
    
    
#    print('**********')
#    print(np.size(image_freq_out[:,count]))
#    print(np.size(image_freq.numpy()))

    image_true_out[:,count] = image_true.numpy()
    image_time_out[:,count] = image_time.numpy()
    image_freq_out[:,count] = image_freq.numpy()




        
    count += 1          
    
    
    
    
    if (count == 5):
        label_array = label_array[:].astype(int)
        tissue = (classes[label_array[0][0]],classes[label_array[1][0]],classes[label_array[2][0]],classes[label_array[3][0]],classes[label_array[4][0]])
        print(tissue)


np.savetxt('paper/plot_csv/'+length+'_image_true.csv', image_true_out)
np.savetxt('paper/plot_csv/'+length+'_image_time.csv', image_time_out)
np.savetxt('paper/plot_csv/'+length+'_image_freq.csv', image_freq_out)
np.savetxt('paper/plot_csv/'+length+'_freq.csv', Freq)
np.savetxt('paper/plot_csv/'+length+'_time.csv', time)







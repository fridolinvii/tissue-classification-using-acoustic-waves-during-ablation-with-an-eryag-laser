import os
import torch as th
#import scipy.io
import numpy as np
import scipy.fftpack as ft
import scipy.signal as filt
import matplotlib.pyplot as plt
#from scipy.stats import norm
#from scipy.optimize import curve_fit
#from audtorch.metrics.functional import pearsonr
import torch.nn.functional as F
import math
from sklearn.manifold import TSNE

# import tkinter



#####################################################################################
#####################################################################################
def save_checkpoint(state, path, filename):
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=False)
    th.save(state, os.path.join(path, filename))




class loss_function():
    CrossEntropyLoss= th.nn.CrossEntropyLoss(reduction='mean')
    MeanSquareLoss = th.nn.MSELoss()



#####################################################################################
#####################################################################################


## programm the traing part of the network ##
def train_network(model,optimizer, loss,device,training_generator,args,params,data):


      model = model.train()

      loss_value_mean = 0.
      train_counter = 0
      for spec, target_label, holes, day in training_generator:


            if spec.shape[0]>1:


                if train_counter == 0:
                    batch_size = target_label.size(0)



                input = spec.to(device)
                label = target_label.to(device)
                holes = holes.to(device)
                day = day.to(device)

                ################################################################################################
                # input = th.tensor(input, requires_grad=True)



                ## Classification loss
                optimizer.zero_grad()
                model.zero_grad()
                net_siamese, x_holse, x_day, _ = model(input)
                loss_value = loss.CrossEntropyLoss(net_siamese,label)
                loss_value.backward()
                optimizer.step()



                loss_value_mean += loss_value*input.shape[0]

                ## print loss value ##
                ## print only ~6 loss values
                if train_counter % np.floor(len(training_generator)/5) == 0:
                   print("loss function at mini-batch iteration {} : {}".format(train_counter, loss_value))
                   print("--------------------------------------------")
                train_counter += 1


      loss_value_mean /= train_counter

      return model, optimizer, loss_value_mean



## programm the traing part of the network ##
def test_network(model,loss,device,args,training_generator_test,test=False):


    with th.no_grad():
        model = model.eval()

        output = th.tensor([]);
        label  = th.tensor([])
        TSNE_output  = th.tensor([])
        Hole  = th.tensor([])
        loss_value_mean = 0
        # cc = 0
        for spec, target_label, hole, day  in training_generator_test:

            input = spec.to(device)
            net_output, _ , _, tsne_output = model(input)

            TSNE_output = th.cat((TSNE_output, tsne_output.to("cpu")),0)
            Hole= th.cat((Hole, hole.to("cpu")),0)
            output  = th.cat((output, net_output.to("cpu")), 0)
            label  = th.cat((label,target_label), 0)

        label = th.tensor(label,dtype=th.long)
        loss_value_mean  = loss.CrossEntropyLoss(output,label)

    
    if test:
        print("TSNE...")

        X_embedded = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3).fit_transform(TSNE_output)
        # plt.plot(X_embedded[:,0],X_embedded[:,1],'*')
        plt.subplot(211)
        label_ = (label-min(label))/(max(label)-min(label))*255
        plt.scatter(X_embedded[:,0],X_embedded[:,1],c=label_,cmap="gist_rainbow",alpha=0.3) #"hsv"
        plt.subplot(212)
        oo = Hole>=0
        print(X_embedded.shape)
        Hole = Hole[oo]/max(Hole[oo])*255
        plt.scatter(X_embedded[oo,0],X_embedded[oo,1],c=Hole, alpha=0.3 ) #,cmap="hsv") # ,alpha=0.3) #"gist_rainbow"

        plt.show()
        

    return output.numpy(), label.numpy(),  loss_value_mean




#####################################################################################
#####################################################################################
## choose optimizer ##
def choose_optimizer(args,params,model):

    optimizer = th.optim.Adam(model.parameters(), lr=params['learning_rate'] ) #, amsgrad=args.amsgrad) #,weight_decay=params['L2_reguralization']) #additional parameter? amsgrad?

    return optimizer



def fittToFrequency(Freq,P, FrequencyRange):

    f1 = Freq >= FrequencyRange[0]
    f2 = Freq <= FrequencyRange[1]

    f  = f1*f2

    P = P[:,f]
    Freq = Freq[f]

    return P, Freq





def printParams(params):
    print('params = {')
    for key,value in params.items():
	    print('"'+key+'" :', value, ',')
    print('}')

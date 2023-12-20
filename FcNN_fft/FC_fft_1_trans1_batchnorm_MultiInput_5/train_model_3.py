import pathlib
import sys
import time

import numpy as np
import torch as th
from torch.utils import data

from data_loader import data_manager as dm
from model import models as m
from parameter import run_args_3 as run_args
from utils import utils
from data import get_params as gp
import random


def print_params(params):
    print(params)
    


def train(args, epoch_start, state):
    if epoch_start > 0:
        params = state['params']
    else:
        params =  gp.get_params();




    
 
    print_params(params)
    print(args.maxepoch)
    ######################################################################################################
    ######################################################################################################
    ## load data with Data Manager ##
    print("Load data...")

    data_train = dm.LoadData(args.train_interval)
    args.NumHoles = data_train.NumHoles
    args.NumDays = data_train.NumDays
    data_validate = dm.LoadData(args.validate_interval)
#    data_train = data_validate 

    data_manager = dm.DataManager(args=args, params=params, data=data_train, train=True)
    data_manager_validate = dm.DataManager(args=args, params=params, data=data_validate, train=False)


    print("Done!\n")


    # Parameters
    train_params = {'batch_size': params['batch_size'],
                    'shuffle': True,
                    'num_workers': args.nb_workers,
                    'pin_memory': False}
    validate_params = {'batch_size': params['batch_size'],
                       'shuffle': False,
                       'num_workers': args.nb_workers,
                       'pin_memory': False}


    training_generator_validate = data.DataLoader(data_manager_validate, **validate_params)

    ######################################################################################################
    ######################################################################################################
    ## Model ##

    seed = 3407
    random.seed(seed)
    th.manual_seed(seed)
    np.random.seed(seed)

    training_generator = data.DataLoader(data_manager, **train_params)


    device = th.device("cuda:" + str(args.gpu_id) if th.cuda.is_available() else "cpu")
    image, _ , _, _ = data_manager_validate[0]
    model = m.ParametricFCModel(image, args, params)
    

    #################################################################################################33
    #################################################################################################33
    ## Check if model is valid"

    print(device)
    print(model)

    if model.error is True:
        print("Error: Change parameters!")
        print("*************************************")
    else:

        #################################################################################################
        #################################################################################################
        ## load network and initilize parameters ##
        if epoch_start > 0:
            ## Use old Network: load parameters and weights of network ##
            model.load_state_dict(state['network'])
            best_mean_loss = state['best_mean_loss']
            loss = state['loss']
        else:
            ## Use New Network: initialize parameters of network ##
            best_mean_loss = 0 #1e10
            # loss function
            loss = utils.loss_function()

        ## optimizer ##
        optimizer  = utils.choose_optimizer(args=args, params=params, model=model)

        #################################################################################################33
        #################################################################################################33

        model.to(device)
        train_counter = 0


        for epoch in range(epoch_start, args.maxepoch):

            print("****************************************************")
            print("************* epoch ", epoch, "***************************")
            print("****************************************************")
            #### Train ####
            model, optimizer, loss_value_mean = utils.train_network(model, optimizer, loss, device, training_generator,args,params,data_train)
            print("mean loss: {}".format(loss_value_mean))

            #### Validate after each epoch ####
            print("***************************")
            print("********* Validate ************")
            print("***************************")
            output, label, mean_loss_overall = utils.test_network(model, loss, device, args, training_generator_validate)
            print("mean loss: {}".format(mean_loss_overall))

            ##################################################################################
            ##################################################################################

            ## Do a weighted accuracy -- makes sense, if distribution of classes is unequal ##


            # find weight of each class
            weight = np.zeros((len(args.classes),1))
            confusion_matrix = np.zeros((len(args.classes),len(args.classes)))

            # create confusion matrix
            for i in range(len(label)):
                confusion_matrix[np.argmax(output[i,:]),label[i]] += 1

            class_accuracy = np.zeros((len(args.classes),1))
            for i in range(len(args.classes)):
                weight = np.sum(label==i)
                class_accuracy[i] = confusion_matrix[i,i]/weight






            loss_value_mean_validate = np.mean(class_accuracy)
            ## save and print ##
            print('************* Validate DATA *******************')
            print('Accuracy of the network on the Validation images: %.5f %%' % (loss_value_mean_validate*100))
            np.savetxt(args.path_to_folder + "/results/model_3/class_predict_last", confusion_matrix.astype(int), fmt='%d')

            ## print confusion matrix ##
            for i in range(len(args.classes)):
                print('Accuracy of %5s : %.5f %%' % (
                args.classes[i], 100 * class_accuracy[i]))

            ##################################################################################
            ##################################################################################

            ## save last and best state ##

            ## we use mean_accuracy as measurment ##
            state = {
                'train_counter': train_counter,
                'args': args,
                'network': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'best_mean_loss': best_mean_loss,
                'params': params,
                'loss': loss,
                }

            if best_mean_loss < loss_value_mean_validate:
                best_epoch = epoch
                best_mean_loss = loss_value_mean_validate 
                state['best_mean_loss'] = best_mean_loss
                utils.save_checkpoint(state, args.path_to_folder+'/results/model_3', "model_best.pt")
                np.savetxt(args.path_to_folder+'/results/model_3' + "/class_predict_best",confusion_matrix.astype(int), fmt='%d')

            utils.save_checkpoint(state, args.path_to_folder+'results/model_3', "model_last.pt")


            
            if epoch - best_epoch > 10:
                break

        ##################################################################################
            ##################################################################################


if __name__ == "__main__":

    input_args = run_args.parser.parse_args()

    if input_args.mode == 'continue':
        print("Continue Training")
        path = args.path_to_folder+'/results/model_3/model_last.pt'
        state = th.load(path)
        args = state['args']
        epoch_start = state['epoch'] + 1
    elif input_args.mode == 'restart':
        print("Restart Training")
        args = input_args
        epoch_start = 0
        state = 0
    else:
        raise NotImplementedError('unknown mode')

    ## create folder, if they don't exist ##
    pathlib.Path(args.path_to_folder+'/results/output_3').mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.path_to_folder+'/results/model_3').mkdir(parents=True, exist_ok=True)

    ## write all the output in log file ##
    if input_args.logfile is True:
        time_string = time.strftime("%Y%m%d-%H%M%S")
        log_file_path = args.path_to_folder+'/results/output_3/output_{}.log'.format(time_string)
        print("Check log file in: ")
        print(log_file_path)
        sys.stdout = open(log_file_path, 'w')
    else:
        print("No log file, only print to console")

    train(args, epoch_start, state)

import pathlib
import sys
import time

import numpy as np
import torch as th
from torch.utils import data

from data_loader import data_manager as dm
from model import models as m
from parameter import run_args_0 as run_args
from utils import utils
from data import get_params as gp
import random


import optuna


def objective(trial):


    epoch_start = 0
    state = 0
    args.maxepoch = 100

    
    # learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    # Integer parameter
    learning_rate = trial.suggest_int("learning_rate", 1, 5)
    batch_size = trial.suggest_int("batch_size", 4, 10)

    n_fc_layers = trial.suggest_int("n_fc_layers", 1, 5)
    neuron = trial.suggest_int("neuron_per_layer", 4, 10)
    dropout = trial.suggest_float("dropout", 0., 0.5)


    n_cnn_layers = trial.suggest_int("n_cnn_layers", 1, 5)
    cnn_kernel_size = trial.suggest_int("cnn_kernel_size", 1, 8)
    max_pooling_kernel_size = trial.suggest_int("max_pooling_kernel_size", 1, 7)
    cnn_output_channels = trial.suggest_int("cnn_output_channels", 1, 10)



    tsne_layer = 0 # trial.suggest_int("tsne_layer", 0, n_fc_layers-1)
 
    
    params =  gp.get_params();
    params['learning_rate'] = 10.**(-learning_rate)
    params['batch_size'] = 2**batch_size

    params['n_fc_layers'] = n_fc_layers
    params['neurons'] = 2**neuron
    params['dropout'] = dropout

    params['tsne_layer'] = tsne_layer

    params["n_cnn_layers"] = n_cnn_layers
    params["cnn_kernel_size"] = 2**cnn_kernel_size
    params["max_pooling_kernel_size"] = 2**max_pooling_kernel_size
    params["cnn_output_channels"] = cnn_output_channels



    ######################################################################################################
    ######################################################################################################

    data_train = args.data_train
    data_validate = args.data_validate
    data_test = args.data_test

    ######################################################################################################
    ######################################################################################################



    

    error = train(args, epoch_start, params, data_train, data_validate, data_test)
    print("*****************************")
    print(args.best_error_all)

    return error





def train(args, epoch_start, params, data_train, data_validate, data_test):

     
    utils.printParams(params)


    data_manager = dm.DataManager(args=args, params=params, data=data_train, train=True)
    data_manager_validate = dm.DataManager(args=args, params=params, data=data_validate, train=False)
    data_manager_test = dm.DataManager(args=args, params=params, data=data_test, train=False)

    # Parameters
    train_params = {'batch_size': params['batch_size'],
                    'shuffle': True,
                    'num_workers': args.nb_workers,
                    'pin_memory': False}
    validate_params = {'batch_size': params['batch_size'],
                       'shuffle': False,
                       'num_workers': args.nb_workers,
                       'pin_memory': False}
    test_params = {'batch_size': params['batch_size'],
                       'shuffle': False,
                       'num_workers': args.nb_workers,
                       'pin_memory': False}


    training_generator_validate = data.DataLoader(data_manager_validate, **validate_params)
    training_generator_test = data.DataLoader(data_manager_test, **test_params)

    ######################################################################################################
    ######################################################################################################
    ## Model ##

    # data
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

        ## Loss function ##
        loss = utils.loss_function()

        ## optimizer ##
        optimizer  = utils.choose_optimizer(args=args, params=params, model=model)

        #################################################################################################33
        #################################################################################################33

        model.to(device)
        train_counter = 0

        best_error = 1
        best_epoch = 0
        test_accuracy = 0

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

            ## print confusion matrix ##
            for i in range(len(args.classes)):
                print('Accuracy of %5s : %.5f %%' % (
                args.classes[i], 100 * class_accuracy[i]))


            if best_error > 1-loss_value_mean_validate:
                best_error = 1-loss_value_mean_validate
                best_epoch = epoch
                #### Test after each best epoch ####
                print("***************************")
                print("********* Test ************")
                print("***************************")
                output, label, mean_loss_overall = utils.test_network(model, loss, device, args, training_generator_test)
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

                test_accuracy = 1-np.mean(class_accuracy)
                print('Test accuracy of the network: %.5f ' % ((1-test_accuracy)*100))


            if epoch - best_epoch > 10:
                break


            print('Best accuracy of the network on the Validation images: %.5f ' % ((1-best_error)*100))
   
            
            ##################################################################################
            ##################################################################################

        print('Best Error of the Network: %.5f' % (best_error))
        



        return test_accuracy

def print_best_callback(study, trial):
    print(f"Best value: {study.best_value}, Best params: {study.best_trial.params}")

if __name__ == "__main__":

    args = run_args.parser.parse_args()
    args.best_error_all = 1
    ## write all the output in log file ##
    ## create folder, if they don't exist ##
    pathlib.Path(args.path_to_folder+'/results/optuna_0').mkdir(parents=True, exist_ok=True)

    if args.logfile is True:
        time_string = time.strftime("%Y%m%d-%H%M%S")
        log_file_path = args.path_to_folder+'/results/optuna_0/optuna_{}.log'.format(time_string)
        print("Check log file in: ")
        print(log_file_path)
        sys.stdout = open(log_file_path, 'w')
    else:
        print("No log file, only print to console")


    ## load data with Data Manager ##
    print("Load data...")
    args.data_train = dm.LoadData(args.train_interval)
    args.NumHoles = args.data_train.NumHoles
    args.NumDays = args.data_train.NumDays
    args.data_validate = dm.LoadData(args.validate_interval)
    args.data_test = dm.LoadData(args.test_interval)
#    data_train = data_validate 
    print("Done!\n")


    study = optuna.create_study()
    study.optimize(objective, n_trials=100, callbacks=[print_best_callback])
    best_params = study.best_params

    print("Best value: {} (params: {} )".format(study.best_value, study.best_params))
    print(log_file_path)


    
    
    

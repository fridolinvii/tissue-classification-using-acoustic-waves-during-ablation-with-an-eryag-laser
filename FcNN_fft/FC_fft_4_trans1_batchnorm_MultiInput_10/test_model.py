import torch as th
import numpy as np
from data_loader import data_manager as dm
from torch.utils import data
from model import models as m
from utils import utils
from parameter import run_args_0 as run_args
import pathlib

def infer_and_evaluate(args, state, dd, TSNE):
    params = state['params']
    ## loss function
    loss = state['loss']


    ######################################################################################################
    ######################################################################################################
    ## load data with Data Manager ##

    data_train = dm.LoadData(dd)
    print("Done!\n")

    data_manager_test = dm.DataManager(args=args, params=params, data=data_train, train=False)
    params_test = {'batch_size': params['batch_size'],
                   'shuffle': False,
                   'num_workers': args.nb_workers,
                   'pin_memory': True}

    training_generator_test = data.DataLoader(data_manager_test, **params_test)

    ######################################################################################################
    ######################################################################################################
    ## Model ##

    image, _ , _ , _ = data_manager_test[0]
    model = m.ParametricFCModel(image, args, params)

    print(model)
    model.load_state_dict(state['network'])

    print("******************** CUDA ***********************")
    device = th.device("cuda:" + str(args.gpu_id) if th.cuda.is_available() else "cpu")
    model.to(device)
    print(device)

    ##########################################################################################################
    ##########################################################################################################
    #### Evaluate ####

#    output_var, _, output, loss_value_mean = utils.test_network(model, loss, device, args
#                                                                                     , training_generator_test, test=True)

    output, label,  loss_value_mean = utils.test_network(model, loss, device, args, training_generator_test, test=TSNE)



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


    print(confusion_matrix)

    loss_value_mean_validate = np.mean(class_accuracy)
    ## save and print ##
    print('************* Validate DATA *******************')
    print('Accuracy of the network on the Validation images: %.5f %%' % (loss_value_mean_validate*100))
    np.savetxt(args.path_to_folder + "/results/model_"+model_+"/class_predict_last", confusion_matrix.astype(int), fmt='%d')

    ## print confusion matrix ##
    for i in range(len(args.classes)):
        print('Accuracy of %5s : %.5f %%' % (
        args.classes[i], 100 * class_accuracy[i]))

    np.savetxt(args.path_to_folder + "/results/model_"+model_+"/class_predict_last", confusion_matrix.astype(int), fmt='%d')

    print("mean loss: {}".format(loss_value_mean_validate*100))

    ##########################################################################################################
    ##########################################################################################################


    pathlib.Path(args.path_to_folder+"/results/gradcam_"+model_).mkdir(parents=True, exist_ok=True)
    np.savetxt(args.path_to_folder + "/results/gradcam_"+model_+"/output.txt", output.astype(float), fmt='%f')
    np.savetxt(args.path_to_folder + "/results/gradcam_"+model_+"/label.txt", label.astype(float), fmt='%d')












    ##########################################################################################################
    ##########################################################################################################


if __name__ == "__main__":

    args = run_args.parser.parse_args()
#     dataset_for_inference = args.infer_data

#     if dataset_for_inference == 'train':
#         chose_path = 'train'
# #        chose_path = args.traincsv_path
#     elif dataset_for_inference == 'test':
#         chose_path = 'test'
# #        chose_path = args.testcsv_path
#     elif dataset_for_inference == 'validate':
#         chose_path = 'validate'
# #       chose_path = args.validatecsv_path
#     else:
#         raise NotImplementedError



    model_ = input("Select Model: ")
    dd = input("Select Data: ")
    TSNE = input("Press enter to skip TSNE, otherwise enter anything. " )

    if len(TSNE)==0:
        TSNE = False
    else :
        TSNE = True

    checkpoint_to_load = args.infer_model
    if checkpoint_to_load == 'last':
        path = args.path_to_folder+"/results/model_"+model_+"/model_last.pt"
        state = th.load(path)
        args = state['args']
        mod_model = 'last'
    elif checkpoint_to_load == 'best':
        path = args.path_to_folder+"/results/model_"+model_+"/model_best.pt"
        state = th.load(path)
        args = state['args']
        mod_model = 'best'
    else:
        raise NotImplementedError

    # print("Infer and evaluate on {} data, using the {} checkpoint".format(dataset_for_inference, checkpoint_to_load))
    print("change data using the --infer_data flag to train, validate or test")
    print("change checkpoint using the --infer_model flag to last or best")

    epoch_start = state['epoch']
    print("****************************************************************")
    print("************* epoch ", epoch_start, ", ***********************")
    print("****************************************************************")

    # print(chose_path)
    infer_and_evaluate(args, state, dd, TSNE)

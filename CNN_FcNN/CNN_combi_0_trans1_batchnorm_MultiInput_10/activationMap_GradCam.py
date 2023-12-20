import matplotlib.pyplot as plt
import numpy as np
import torch as th
import torch.nn.functional as F
import pathlib

from scipy.ndimage import gaussian_filter
# from skimage import transform
from torch.utils import data

from data_loader import data_manager as dm
from model import models as m
from parameter import run_args 
from utils import utils

from scipy import interpolate

def grad_cam(args_fixed, state, chose_path):
    params = state['params']

    ## loss function
    loss = state['loss']

    ######################################################################################################
    ######################################################################################################
    ## load data with Data Manager ##

    print("\nLoad data...")
    if chose_path == 'train':
        print("train")
        interval = args.train_interval
    elif chose_path == 'validate':
        print("validate")
        interval = args.validate_interval
    else:
        print("test")
        interval = args.test_interval


    data_train = dm.LoadData(args=args,interval=interval)

    data_manager_test = dm.DataManager(args=args, params=params, data=data_train, train=False)
    print("Done!\n")


    params_test = {'batch_size': params['batch_size'],
                   'shuffle': False,
                   'num_workers': args.nb_workers,
                   'pin_memory': True}

    training_generator_test = data.DataLoader(data_manager_test, **params_test)

    ######################################################################################################
    ######################################################################################################
    ## Model ##

    # device = th.device("cuda:" + str(args_fixed.gpu_id) if th.cuda.is_available() else "cpu")
    device = "cuda:0"
    image, _ = data_manager_test[0]
    model = m.ParametricConvFCModel(image, args_fixed, params)

    print(model)
    model.load_state_dict(state['network'])

    for mm in model.modules():
        if isinstance(mm, th.nn.BatchNorm1d) or isinstance(mm, th.nn.BatchNorm2d):
            mm.eval()

    ## optimizer ##
    optimizer = utils.choose_optimizer(args=args_fixed, params=params, model=model)

    print("******************** CUDA ***********************")
    model.to(device)
    print(device)

    ##########################################################################################################
    ##########################################################################################################
    #### Evaluate ####

    count = 0
    output = th.tensor([]);
    label = th.tensor([]);
    activity_map = th.tensor([]);
#    with th.no_grad():
#        model = model.eval()
    for spec, target_label in training_generator_test:

            optimizer.zero_grad()
            model.zero_grad()
            input = spec.to(device)
            target_label = target_label.to(device)
            net_output = model(input)

            _, predicted = th.max(net_output, 1)

            ## Grad Cam ##

            optimizer.zero_grad()
            model.grad_cam_feature_maps.retain_grad()

            loss_value = loss.CrossEntropyLoss(net_output, target_label)
            loss_value.backward()

            feature_map_gradients_max, _ = th.max(abs(model.grad_cam_feature_maps.grad), -1)
            feature_map_gradients = model.grad_cam_feature_maps.grad

            # for i in range(feature_map_gradients_max.size(0)):
            #     for j in range(feature_map_gradients_max.size(1)):
            #         feature_map_gradients[i,j,:] /= feature_map_gradients_max[i,j]

            feature_map_gradients = th.sum(F.relu(feature_map_gradients), 1)

            plt_grad = feature_map_gradients[0,:].to("cpu").numpy();
            plt_grad = plt_grad/(max(plt_grad)+1e-10)
            plt_spec = input[0,0,:].to("cpu").numpy();
            plt_spec = plt_spec/(max(abs(plt_spec))+1e-10)


            x = np.linspace(0,1,len(plt_grad))
            xnew = np.linspace(0,1,len(plt_spec))
            f = interpolate.interp1d(x,plt_grad)
            plt_grad = f(xnew)

            label_ = [target_label[0].to("cpu").numpy(),predicted[0].to("cpu").numpy()]

            plt.plot(plt_grad)
            plt.plot(plt_spec)
            plt.title(label_)
            plt.show()

#            output = th.cat((output, net_output.detach().to("cpu")), 0)
#            label = th.cat((label, target_label.detach().to("cpu")), 0)
            activity_map = th.cat((activity_map,feature_map_gradients.detach().to("cpu")),0)


#    np.savetxt(args_fixed.path_to_folder + "/results/gradcam/output.txt", output.numpy().astype(float), fmt='%f')
#    np.savetxt(args_fixed.path_to_folder + "/results/gradcam/label.txt", label.numpy().astype(float), fmt='%f')
    np.savetxt(args_fixed.path_to_folder + "/results/gradcam/activity_map.txt", activity_map.numpy().astype(float), fmt='%f')



    ##########################################################################################################
    ##########################################################################################################


if __name__ == "__main__":


    args = run_args.parser.parse_args()
    dataset_for_inference = args.infer_data

    pathlib.Path(args.path_to_folder+'/results/gradcam').mkdir(parents=True, exist_ok=True)

    if dataset_for_inference == 'train':
        chose_path = 'train'
#        chose_path = args.traincsv_path
    elif dataset_for_inference == 'test':
        chose_path = 'test'
#        chose_path = args.testcsv_path
    elif dataset_for_inference == 'validate':
        chose_path = 'validate'
#       chose_path = args.validatecsv_path
    else:
        raise NotImplementedError


    checkpoint_to_load = args.infer_model
    if checkpoint_to_load == 'last':
        path = args.path_to_folder+'/results/model/model_last.pt'
        state = th.load(path)
        args = state['args']
        mod_model = 'last'
    elif checkpoint_to_load == 'best':
        path = args.path_to_folder+'/results/model/model_best.pt'
        state = th.load(path)
        args = state['args']
        mod_model = 'best'
    else:
        raise NotImplementedError

    print("Infer and evaluate on {} data, using the {} checkpoint".format(dataset_for_inference, checkpoint_to_load))
    print("change data using the --infer_data flag to train, validate or test")
    print("change checkpoint using the --infer_model flag to last or best")

    epoch_start = state['epoch']
    print("****************************************************************")
    print("************* epoch ", epoch_start, ", ***********************")
    print("************* path ", chose_path, ", ***********************")
    print("****************************************************************")

    print(chose_path)
    grad_cam(args, state, chose_path)


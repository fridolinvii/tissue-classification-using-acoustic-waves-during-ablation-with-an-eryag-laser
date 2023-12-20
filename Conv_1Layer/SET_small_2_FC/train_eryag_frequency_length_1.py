__author__ = "Carlo Seppi"
__copyright__ = "Copyright (C) 2019 Center for medical Image Analysis and Navigation"
__email__ = "carlo.seppi@unibas.ch"


import torch as th
import numpy as np
# import visdom as vis


from data_loader import data_manager as dm
from torch.utils import data
from model import models as m
from parameter import parameter_eryag_frequency_length_1 as param
from utils import utils


def train(args,epoch_start,state):
    gpu_id = args.gpu_id

    th.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = th.device("cuda:" + str(gpu_id))

    data_train = dm.LoadData(args.train_interval, args.path)
    data_validate = dm.LoadData(args.validate_interval, args.path)

    data_manager = dm.DataManager_Frequency(data_train, args.samplerate, args.frequencyrange)
    data_manager_validate = dm.DataManager_Frequency(data_validate, args.samplerate, args.frequencyrange)

    # Parameters
    params = {'batch_size': args.batch_size,
              'shuffle': True,
              'num_workers': 10, #args.nb_workers,
              'pin_memory': True}

    training_generator = data.DataLoader(data_manager, **params)
    training_generator_validate = data.DataLoader(data_manager_validate, **params)

    # create model
    if args.model == "cnn_frequency":
        print(data_manager.input_size)
        model = m.ConvClasi_frequency_length_1(data_manager.input_size, num_classes=len(args.classes))

    best_mean_accuracy = 0    
    if epoch_start > 0:
        model.load_state_dict(state['network'])    
        best_mean_accuracy = state['best_mean_accuracy']
        
    print("******************** CUDA ***********************")
    device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
    model.to(device)
    print(device)   
    


#    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
#    params = sum([np.prod(p.size()) for p in model_parameters])
#    print("number of parameters model", params)



    if args.optimizer == 'RMSprop':
        optimizer = th.optim.RMSprop(model.parameters(), lr=args.lr)
    elif args.optimizer == 'Adam':
        optimizer = th.optim.Adam(model.parameters(), lr=args.lr, amsgrad=args.amsgrad)
    elif args.optimizer == 'Rprop':
        optimizer = th.optim.Rprop(model.parameters(), lr=args.lr)
    elif args.optimizer == 'SGD':
        optimizer = th.optim.SGD(model.parameters(), lr=args.lr)

    # loss function
    loss = th.nn.CrossEntropyLoss()

#    # define evaluater
#    evaluater = Evaluater(args.eval_path, args, device=device)


    train_counter = 0
    eval_counter = 0
    loss_plot = None
    eval_plot = None
    
    alpha = 6
    for epoch in range(epoch_start,args.num_epoch):
    
    
        model = model.train()
    
    
        file = open(args.epoch_path+"/epoch%d.txt" %(epoch), "w")
    
        print("****************************************************************")
        print("************* epoch ",epoch,"***********************")
        print("****************************************************************")
    
    
        epoch_counter = 0
        for spec, target_label in training_generator:
        
        
            epoch_counter += 1;

            optimizer.zero_grad()
            model.zero_grad()

            spec = spec.unsqueeze(1).to(device)
            spec *= alpha
            target_label = target_label.to(device)


            

            net_output = model(spec)
            loss_value = loss(net_output, target_label)
            loss_value.backward()

            
            if epoch_counter % 1000 == 1:    # print every 10000 mini-batches
                print("loss function", train_counter, loss_value.item())


            
            
            optimizer.step()

            loss_value_ = np.column_stack(np.array([loss_value.item()]))

#            if loss_plot is None:
#                opts = dict(title="loss_value ", width=1000, height=500, showlegend=True,
#                            legend=['CEL loss'])
#                loss_plot = viz.line(X=np.column_stack(np.ones(1) * train_counter), Y=loss_value_, opts=opts)
#            else:
#                loss_plot = viz.line(X=np.column_stack(np.ones(1) * train_counter), Y=loss_value_, win=loss_plot,
#                                     update='append')



            train_counter += 1
            
            
            
            
        ##########################################################################################################    
        #### Evaluate ####
            
            
        correct = 0
        total = 0
        
        class_correct = list(0. for i in range(len(args.classes)))
        class_total = list(0. for i in range(len(args.classes)))
        with th.no_grad():
            model = model.eval()
            for spec, target_label in training_generator_validate:
            
            
                inputs = spec.unsqueeze(1).to(device)
                inputs *= alpha
                labels = target_label.to(device)
                output = model(inputs)
            
               
                _, predicted = th.max(output.data, 1)
                
                c = (predicted == labels).squeeze()
                for i in range(len(labels)):
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1
                        
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()



        print('************* TEST DATA *******************')
        print('Accuracy of the network on the test images: %.5f %%' % (
            100 * correct / total))
        file.write('************* TEST1 DATA ******************* \n')
        file.write('Accuracy of the network on the test images: %.5f %% \n' % (
            100 * correct / total))


        for i in range(len(args.classes)):
            print('Accuracy of %5s : %.5f %%' % (
                args.classes[i], 100 * class_correct[i] / class_total[i]))
            file.write('Accuracy of %5s : %.5f %% \n' % (
                args.classes[i], 100 * class_correct[i] / class_total[i]))

        file.close()
        
        mean_accuracy_overall = (100 * correct / total)

        state = {
                 'train_counter': train_counter,
                 'args': args,
                 'network': model.state_dict(),
                 'optimizer': optimizer.state_dict(),
                 'epoch': epoch,
                 'best_mean_accuracy': best_mean_accuracy}


        if best_mean_accuracy < mean_accuracy_overall:
            best_mean_accuracy = mean_accuracy_overall
            state['best_mean_accuracy'] = best_mean_accuracy
            utils.save_checkpoint(state, args.result_path, "model_best.pt")
            
            

        utils.save_checkpoint(state, args.result_path, "model_last.pt")
        
        if best_mean_accuracy > mean_accuracy_overall:
            acc_counter += 1
            if acc_counter > 10:
                print(f'Early stopping: Best accuracy: {best_mean_accuracy:.6f} -- Current accuracy: {mean_accuracy_overall:.6f}')
                break
            else:
                print(f'Accuracy not improved for {acc_counter} epochs')
        else:
            acc_counter = 0
            print(f'Accuracy improved ({best_mean_accuracy:.6f} --> {mean_accuracy_overall:.6f}). Saving model ...')


if __name__ == "__main__":

    mod = 0
    if (mod==1):
        path = 'data/ErYAG/frequency_results_length_1/model_last.pt'
        state = th.load(path, map_location="cpu")
        args = state['args']
        epoch_start = state['epoch']+1
        
    else  :
        args = param.parser.parse_args()
        epoch_start = 0;
        state = 0;
        
    
    train(args,epoch_start,state)

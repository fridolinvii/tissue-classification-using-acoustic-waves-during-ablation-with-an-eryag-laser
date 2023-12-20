__author__ = "Carlo Seppi"
__copyright__ = "Copyright (C) 2019 Center for medical Image Analysis and Navigation"
__email__ = "carlo.seppi@unibas.ch"

import argparse

parser = argparse.ArgumentParser()





parser.add_argument(
    '--frequencyrange',
    default=[1e5, 8e5],
    help='Frequency Range to evaluate'
)

parser.add_argument(
    '--samplerate',
    default=7.8125e6,
    help='Sample Rate'
)


parser.add_argument(
    '--train-interval', dest='train_interval',
    default='495016',
    help='path of the csv test file'
)
parser.add_argument(
    '--validate-interval', dest='validate_interval',
    default='27',
    help='path of the csv test file'
)
parser.add_argument(
    '--test-interval', dest='test_interval',
    default='38',
    help='path of the csv test file'
)




parser.add_argument(
    '--path',
    default='data/TissueDifferentation_2022/Measurment_22-24.11.22/matrix_GradCam_old/time_matrix_3000/', #/home/carlo.seppi/Tissue_Differentiation_Microphone_25.07.2019/data/ErYAG/time',
    help='path of the data'
)



parser.add_argument(
    '--result-path',
    default='data/ErYAG/time_results_length_1_3000',
    help='path of the evaluation data'
)



parser.add_argument(
    '--traincsv-path',
    default='data/Train.csv',
    help='path of the csv train file'
)

parser.add_argument(
    '--testcsv-path',
    default='data/Test.csv',
    help='path of the csv test file'
)



parser.add_argument(
    '--evaluatecsv-path',
    default='data/Evaluate.csv',
    help='path of the csv Evaluate file'
)


parser.add_argument(
    '--classes',
    default=('HardBone','SoftBone','Fat','Skin','Muscle'),
    help='classes'
)



parser.add_argument(
    '--mult',
    default=3,   #maximal value
    help='multipli images with this factor'
)





parser.add_argument(
    '--epoch-path',
    default='data/ErYAG/epoch_time_length_1_3000/',
    help='where the epoch file is saved'
)






#########################################################################

parser.add_argument(
    '--port',
    type=int,
    default=8097,
    help='port for the visdom server'
)

parser.add_argument(
    '--seed',
    type=int,
    default=123456789,
    metavar='S',
    help='random seed (default: 1)'
)

parser.add_argument(
    '--gpu-id',
    type=int,
    default=1,
    help='gpu id if set to -1 then use cpu'
)

parser.add_argument(
    '--batch-size',
    type=int,
    default=16,
    help='batch size for the training'
)


parser.add_argument(
    '--nb-workers',
    type=int,
    default=1,
    help='number of workers for the data loader'
)


parser.add_argument(
    '--model',
    default="cnn_frequency",
    help='model type'
)


parser.add_argument(
    '--optimizer',
    default='Adam',
    metavar='OPT',
    help='shares optimizer choice of Adam, SGD or RMSprop'
)



parser.add_argument(
    '--lr',
    type=float,
    default=0.001, #0.0001,
    metavar='LR',
    help='learning rate (default: 0.0001)'
)


parser.add_argument(
    '--amsgrad',
    default=True,
    type=lambda x: (str(x).lower()) == 'true',
    metavar='AM',
    help='Adam optimizer amsgrad parameter'
)



parser.add_argument(
    '--num-epoch',
    type=int,
    default=500,
    help='batch size for the training'
)



parser.add_argument(
    '--eval-iter',
    type=int,
    default=10,
    help='number of iterations between two evaluations'
)

#parser.add_argument(
#    '--dropout',
#    type=float,
#    default=0.0,
#    help='dropout value'
#)



#parser.add_argument(
#    '--save-model-iter',
#    type=int,
#    default=1000,
#    help='save the current state of the model and the optimzer'
#)





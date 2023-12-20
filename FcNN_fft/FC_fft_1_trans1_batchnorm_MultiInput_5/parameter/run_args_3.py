__author__ = "Carlo Seppi, Eva Schnider"
__copyright__ = "Copyright (C) 2019 Center for medical Image Analysis and Navigation"
__email__ = "carlo.seppi@unibas.ch"

import argparse

parser = argparse.ArgumentParser(description="Parameters for 1D Convolutional Nets")


parser.add_argument(
    '--path-to-folder',
    default='FC_fft_1_trans1_batchnorm_MultiInput2/',
    help='name the folder'
)


parser.add_argument(
    '--frequencyrange',
    default=[6.66e5, 1e6],
    help='Frequency Range to evaluate'
)

# parser.add_argument(
#     '--samplerate',
#     default=7.8125e6,
#     help='Sample Rate'
# )


# parser.add_argument(
#     '--freqfilter',
#     default=False,
#     help='Frequency filter Data'
# )



parser.add_argument(
    '--train-interval', dest='train_interval',
    default='273849',
    help='path of the csv test file'
)
parser.add_argument(
    '--validate-interval', dest='validate_interval',
    default='05',
    help='path of the csv test file'
)
parser.add_argument(
    '--test-interval', dest='test_interval',
    default='16',
    help='path of the csv test file'
)

parser.add_argument(
    '--classes', dest='classes',
    default=['HardBone','SoftBone','Fat','Skin','Muscle'],
    help = 'what classes it should have'
)

# parser.add_argument(
#     '--area-of-interest', dest='area_of_interest',
#     default=[0,-1], #if -1 it takes the hole array
#     help = '[minimum_frame, maximum_frame], maximum_frame = -1, takes the array until the end'
# )


parser.add_argument(
    "-m", action="store", dest="mode", default="restart", type=str, choices=["restart", "continue"],
    help="Choose whether to restart the training from scratch, or to continue from model_last."
)

parser.add_argument(
    "--infer-model", action="store", dest="infer_model", default="last", type=str, choices=["last", "best"],
    help="Choose which data partition to use for inference."
)

parser.add_argument(
    "--logfile", action="store_true", dest="logfile", default=False,
    help="Add this flag if you want to redirect the console output to a file."
)



parser.add_argument(
    '--nb-workers', dest='nb_workers',
    type=int,
    default=8,
    help='number of workers for the data loader'
)



# parser.add_argument(
#     '--frame-rate',
#     default=7.8125e6,
#     help='how many frame one window has'
# )



# parser.add_argument(
#     '--path',
#     default='data/',
#     help='path of the data'
# )

# parser.add_argument(
#     '--data-path', dest='data_path',
#     default='../data/20220310-11-LabeledData_3classes.csv',
#     help='path of the csv train file'
# )










parser.add_argument(
    '--gpu-id', dest='gpu_id',
    type=int,
    default=0,
    help='gpu id if set to -1 then use cpu'
)

parser.add_argument(
    '--maxepoch',
    default=500,
    help='Number of Epochs used in training'
)


# parser.add_argument(
#     '--amsgrad',
#     default=True,
#     type=lambda x: (str(x).lower()) == 'true',
#     metavar='AM',
#     help='Adam optimizer amsgrad parameter'
# )

# parser.add_argument(
#     '--fcinputsize',
#     default=[30, 15000],
#     help='Min. and Max. number of Neurons allowed'
# )

# #########################################################################

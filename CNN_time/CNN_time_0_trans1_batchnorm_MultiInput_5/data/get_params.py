__author__ = "Carlo Seppi"
__copyright__ = "Copyright (C) 2022 Center for medical Image Analysis and Navigation"
__email__ = "carlo.seppi@unibas.ch"


# put here the params from hyperparametersearch


def get_params():


    params = {
"n_cnn_layers" : 4 ,
"cnn_kernel_size" : 128 ,
"max_pooling_kernel_size" : 8 ,
"cnn_output_channels" : 8 ,
"n_fc_layers" : 1 ,
"neurons" : 16 ,
"batch_size" : 256 ,
"tsne_layer" : 0 ,
"learning_rate" : 0.0001 ,
"dropout" : 0.3923535702072316 ,
}




    return params;



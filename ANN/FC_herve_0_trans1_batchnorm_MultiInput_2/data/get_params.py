__author__ = "Carlo Seppi"
__copyright__ = "Copyright (C) 2022 Center for medical Image Analysis and Navigation"
__email__ = "carlo.seppi@unibas.ch"


# put here the params from hyperparametersearch


def get_params():

    params = {
"n_fc_layers" : 1 ,
"neurons" : 10 ,
"batch_size" : 32 ,
"tsne_layer" : 0 ,
"learning_rate" : 0.1 ,
"dropout" : 0 ,
"pca_components" : 3 ,
}



    return params;



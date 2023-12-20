# Deep-learning approach for tissue classification using acoustic waves during ablation with an Er:YAG laser
This is the code to the paper [Deep-learning approach for tissue classification using acoustic waves during ablation with an Er:YAG laser](https://doi.org/10.1109/ACCESS.2021.3113055) (IEEE Access 2021).

## ATTENTION
This This manuscript is currently undergoing revision due to issues identified with the data we received.
It was found that some of the training, testing, and validation datasets were scaled relative to each other. 
This scaling inadvertently resulted in an unrealistically high accuracy rate of 100% for our neural network.

Upon recognizing this data-related issue, we conducted a thorough reevaluation by repeating the experiments. 
These revised measurements revealed a significantly lower accuracy rate, falling below 40% – a figure markedly lower than what was initially anticipated.

In response to these findings, we have extensively revised and enhanced our network architecture. 
The improvements made have led to a substantially better-performing model. 
Our optimized network now achieves a significantly higher accuracy rate, exceeding 75%. 
This notable enhancement marks a considerable improvement in both the performance and reliability of our model.


## Network
We will give here an explenation of the initial network with the old data, with the new data, and with the new data and network. 


## Branches Description
* branch *old*: initial code of the paper. 
    * issues with the data (some of the data were scaled relative to each other)
    * unrealistic accuracy of 100% of the neural network
    * published initially in IEEE Access 2021
* branch *old_with_new_data*: repetion of the experiment. And used new data to retrain the neural network
    * accuracy was under 40% 
    * is described in correction of paper
* branch *main*: used the new data and in depth analisis and fine tuning of the neural network
    * best performing network: exceeds 75% accuracy
    * detailed description can be found on the corrected paper


## Branch *old_with_new_data*
We are testing the performance of the old network from the branch *old* with the new data
* `Conv_1Layer`, `RNN`, and `FC` are the three different neural networks which we compare
* data are divided into 10 subsets represented by the numbers `1,...,9,0'
* the data are pared together as follow: `16,27,38,49,50`
* these are used for training, testing, and validation accordingly to the files in parameter
* adapt the path to the desired folder where the data are saved
* `test_eryag_*_length_all.py` is used to test the performance of the network
* `train_eryag_time_length_{0,1}{.py,_3000.py}`
    * `{0,1}`: `0` uses the frequency range 0-1Mhz, `1` uses the frequency range from 0.1-0.8Mhz
    * `{.py,_3000.py}`: `.py` has has an input array of 1000, `_3000.py` input array of 3000
* `results.odt` overview of the results


## Data
Please contact *Prof. Dr. Philippe C. Cattin* (philippe.cattin@unibas.ch) to access the data

The data can be found on lakeFS: https://dbe-lakefs.dbe.unibas.ch/repositories/tissue-classification-acoustic
* branch *old*: some of the data were scaled relative to each other.
* See `compareData.m` to see an examle of the issues of the data for the measurment with the Er:YAG and Nd:YAG data.
* The initial publication has issues with the high accuracy. 
* code for inital publication can be found in the branch *old* on *GitHub* and *GitLab*
* branch *main*: new measurment done on the 22.-24.11.2022.
    * `saveAsMatrix.m` transforms the data into a matrix and saves them in the folder `matrix_all`
    * settings of laser and the measurment of the acoustic signal, see details in Paper corrected paper 
* branch *old_with_new_data*: the data und the folder `matrix_GradCam_old` uses the new measursments (11.2022) with the initial code from the inital paper 
    * code can be found in the branch *old_with_new_data* on *GitHub* and *GitLab*
    * in the folder `matrix_GradCam_old` is the file `cut_to_size_preprocess.m` which transforms the matrix `matrix_all` in the desired format
* branch *remake*: the data can be found under `matrix_pre_3000`
    * code can be found in the *main* branch on *GitHub* and *GitLab*
    * in the folder `matrix_pre_3000` is the file `preProcessing_pre_3000.m` which transforms the matrix `matrix_all` in the desired format






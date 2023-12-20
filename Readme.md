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






## Data
Please contact *Prof. Dr. Philippe C. Cattin* (philippe.cattin@unibas.ch) to access the data
 
### Original data
* We note, these data have issues and should **NOT** be used for further research 
* https://dbe-lakefs.dbe.unibas.ch/repositories/moonstar-slnf/main

### New data
* These are the new measurment an can be used for further research
* https://dbe-lakefs.dbe.unibas.ch/repositories/moonstar-slnf/objects?ref=slnf&path= (Branch slnf)


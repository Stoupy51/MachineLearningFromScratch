
#ifndef __NEURAL_TRAINING_H__
#define __NEURAL_TRAINING_H__

#include "neural_utils.h"

void NeuralNetworkDbackpropagation(NeuralNetworkD *network, double *excepted_output);
void NeuralNetworkDupdateWeightsAndBiases(NeuralNetworkD *network);
void NeuralNetworkDtrain(NeuralNetworkD *network, double *input, double *excepted_output);


#endif


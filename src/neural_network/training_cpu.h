
#ifndef __NEURAL_TRAINING_CPU_H__
#define __NEURAL_TRAINING_CPU_H__

#include "neural_utils.h"

void NeuralNetworkfeedForwardCPUSingleCore(NeuralNetwork *network, nn_type *input);
void NeuralNetworkbackpropagationCPUSingleCore(NeuralNetwork *network, nn_type *excepted_output);

#endif


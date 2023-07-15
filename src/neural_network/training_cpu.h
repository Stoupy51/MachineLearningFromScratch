
#ifndef __NEURAL_TRAINING_CPU_H__
#define __NEURAL_TRAINING_CPU_H__

#include "neural_utils.h"

void NeuralNetworkDfeedForwardCPU(NeuralNetworkD *network, double *input);
void NeuralNetworkDbackpropagationCPU(NeuralNetworkD *network, double *excepted_output);
void NeuralNetworkDupdateWeightsAndBiasesCPU(NeuralNetworkD *network);
void NeuralNetworkDtrainCPU(NeuralNetworkD *network, double *input, double *excepted_output);

#endif


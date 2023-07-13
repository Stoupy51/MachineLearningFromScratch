
#ifndef __NEURAL_TRAINING_H__
#define __NEURAL_TRAINING_H__

#include "neural_utils.h"

void NeuralNetworkDfeedForward(NeuralNetworkD *network, double *input);
void NeuralNetworkDbackpropagation(NeuralNetworkD *network, double *excepted_output);
void NeuralNetworkDupdateWeightsAndBiases(NeuralNetworkD *network);
void NeuralNetworkDtrain(NeuralNetworkD *network, double *input, double *excepted_output);

///// GPU Part /////
int NeuralNetworkDfeedForwardGPU(NeuralNetworkD *network, double *input);
int NeuralNetworkDbackpropagationGPU(NeuralNetworkD *network, double *excepted_output);
int NeuralNetworkDupdateWeightsAndBiasesGPU(NeuralNetworkD *network);
int NeuralNetworkDtrainGPU(NeuralNetworkD *network, double *input, double *excepted_output);

#endif


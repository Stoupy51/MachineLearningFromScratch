
#ifndef __NEURAL_TRAINING_CPU_H__
#define __NEURAL_TRAINING_CPU_H__

#include "neural_utils.h"
#include "../list/image_list.h"

void NeuralNetworkDfeedForwardCPU(NeuralNetworkD *network, double *input);
void NeuralNetworkDbackpropagationCPU(NeuralNetworkD *network, double *excepted_output);
void NeuralNetworkDupdateWeightsAndBiasesCPU(NeuralNetworkD *network);
void NeuralNetworkDtrainCPU(NeuralNetworkD *network, double *input, double *excepted_output);
int NeuralNetworkDtrainFromImageListCPU(NeuralNetworkD *network, img_list_t img_list, image_t excepted_output);

#endif


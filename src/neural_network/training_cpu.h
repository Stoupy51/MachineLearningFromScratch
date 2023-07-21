
#ifndef __NEURAL_TRAINING_CPU_H__
#define __NEURAL_TRAINING_CPU_H__

#include "neural_utils.h"
//#include "../list/image_list.h"

void NeuralNetworkfeedForwardCPU(NeuralNetwork *network, double *input);
void NeuralNetworkbackpropagationCPU(NeuralNetwork *network, double *excepted_output);
void NeuralNetworkupdateWeightsAndBiasesCPU(NeuralNetwork *network);
void NeuralNetworktrainCPU(NeuralNetwork *network, double *input, double *excepted_output);
//int NeuralNetworktrainFromImageListCPU(NeuralNetwork *network, img_list_t img_list, image_t excepted_output);

#endif


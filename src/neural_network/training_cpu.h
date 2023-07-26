
#ifndef __NEURAL_TRAINING_CPU_H__
#define __NEURAL_TRAINING_CPU_H__

#include "neural_utils.h"

void NeuralNetworkFeedForwardCPUSingleCore(NeuralNetwork *network, nn_type *input, nn_type *output);
void NeuralNetworkStartBackPropagationCPUSingleCore(NeuralNetwork *network, nn_type **predicted, nn_type **expected, int batch_size);
void NeuralNetworkFinishBackPropagationCPUSingleCore(NeuralNetwork *network);
void NeuralNetworkUpdateWeightsCPUSingleCore(NeuralNetwork *network);

// Single-core version

int NeuralNetworkTrainCPUSingleCore(
	NeuralNetwork *network,
	nn_type **inputs,
	nn_type **expected,
	int nb_inputs,
	int test_inputs_percentage,
	int batch_size,
	int nb_epochs,
	nn_type error_target,
	int verbose
);



// Multi-core version
void NeuralNetworkFeedForwardCPUMultiCores(NeuralNetwork *network, nn_type *input, nn_type *output);


int NeuralNetworkTrainCPUMultiCores(
	NeuralNetwork *network,
	nn_type **inputs,
	nn_type **expected,
	int nb_inputs,
	int test_inputs_percentage,
	int batch_size,
	int nb_epochs,
	nn_type error_target,
	int verbose
);


#endif


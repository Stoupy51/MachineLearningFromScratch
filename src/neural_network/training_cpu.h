
#ifndef __NEURAL_TRAINING_CPU_H__
#define __NEURAL_TRAINING_CPU_H__

#include "neural_utils.h"

// Single-core version
void NeuralNetworkFeedForwardCPUSingleThread(NeuralNetwork *network, nn_type *input, nn_type *output);
void NeuralNetworkStartBackPropagationCPUSingleThread(NeuralNetwork *network, nn_type **predicted, nn_type **expected, int batch_size);
void NeuralNetworkFinishBackPropagationCPUSingleThread(NeuralNetwork *network);
void NeuralNetworkUpdateWeightsCPUSingleThread(NeuralNetwork *network);
int NeuralNetworkTrainCPUSingleThread(
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
void NeuralNetworkFeedForwardCPUMultiThreads(NeuralNetwork *network, nn_type *input, nn_type *output);
void NeuralNetworkStartBackPropagationCPUMultiThreads(NeuralNetwork *network, nn_type **predicted, nn_type **expected, int batch_size);
void NeuralNetworkFinishBackPropagationCPUMultiThreads(NeuralNetwork *network);
void NeuralNetworkUpdateWeightsCPUMultiThreads(NeuralNetwork *network);
int NeuralNetworkTrainCPUMultiThreads(
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


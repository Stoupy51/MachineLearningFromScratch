
#ifndef __NEURAL_TRAINING_CPU_H__
#define __NEURAL_TRAINING_CPU_H__

#include "neural_utils.h"

void NeuralNetworkFeedForwardCPUSingleCore(
	NeuralNetwork *network,
	nn_type **inputs,
	nn_type **outputs,
	int batch_size
);

void NeuralNetworkBackPropagationCPUSingleCore(
	NeuralNetwork *network,
	nn_type **predicted,
	nn_type **expected,
	int batch_size
);

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

#endif


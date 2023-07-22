
#ifndef __NEURAL_TRAINING_CPU_H__
#define __NEURAL_TRAINING_CPU_H__

#include "neural_utils.h"

void NeuralNetworkFeedForwardCPUSingleCore(
	NeuralNetwork *network,
	nn_type **inputs,
	nn_type **predicted,
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
	int batch_size,
	nn_type **test_inputs,
	nn_type **excepted_tests,
	int nb_test_inputs,
	int nb_epochs,
	nn_type error_target,
	int verbose
);

#endif


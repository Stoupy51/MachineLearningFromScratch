
#ifndef __NEURAL_TRAINING_CPU_H__
#define __NEURAL_TRAINING_CPU_H__

#include "neural_utils.h"

// Single-core version
void FeedForwardCPUSingleThread(NeuralNetwork *network, nn_type *input);
int TrainCPUSingleThread(
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
// TODO


#endif


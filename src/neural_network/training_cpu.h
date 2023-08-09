
#ifndef __NEURAL_TRAINING_CPU_H__
#define __NEURAL_TRAINING_CPU_H__

#include "neural_utils.h"

typedef void (*optimizer_t)(NeuralNetwork *network, nn_type **predictions, nn_type **targets, int batch_size);

// Single-core version

void FeedForwardCPU(NeuralNetwork *network, nn_type *input);
void FeedForwardBatchCPU(NeuralNetwork *network, nn_type **inputs, nn_type **outputs, int batch_size);
void StochasticGradientDescentCPU(NeuralNetwork *network, nn_type **predictions, nn_type **targets, int batch_size);
void shuffleTrainingData(nn_type **inputs, nn_type **target_outputs, int batch_size);
nn_type ComputeCost(NeuralNetwork *network, nn_type **predicted_outputs, nn_type **target_outputs, int batch_size);
void epochCPU(NeuralNetwork *network, nn_type **inputs, nn_type **target, int nb_inputs, int nb_batches, int batch_size, int current_epoch, int nb_epochs, nn_type *current_error, nn_type **test_inputs, nn_type **target_tests, int nb_test_inputs, int verbose, optimizer_t optimizer);

int TrainCPU(
	NeuralNetwork *network,
	nn_type **inputs,
	nn_type **expected,
	int nb_inputs,
	int test_inputs_percentage,
	int batch_size,
	int nb_epochs,
	nn_type error_target,
	int verbose,
	const char *optimizer
);

#endif


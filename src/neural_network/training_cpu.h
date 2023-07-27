
#ifndef __NEURAL_TRAINING_CPU_H__
#define __NEURAL_TRAINING_CPU_H__

#include "neural_utils.h"

// Single-core version
void FeedForwardCPUSingleThread(NeuralNetwork *network, nn_type *input);
void FeedForwardBatchCPUSingleThread(NeuralNetwork *network, nn_type **inputs, nn_type **outputs, int batch_size);
void FeedForwardBatchCPUMultiThreads(NeuralNetwork *network, nn_type **inputs, nn_type **outputs, int batch_size);
void BackpropagationCPUSingleThread(NeuralNetwork *network, nn_type **predicted_outputs, nn_type **target_outputs, int batch_size);
void MiniBatchGradientDescentCPUSingleThread(NeuralNetwork *network, nn_type **inputs, nn_type **target_outputs, int batch_size);
void shuffleTrainingData(nn_type **inputs, nn_type **target_outputs, int batch_size);
nn_type ComputeCostCPUSingleThread(NeuralNetwork *network, nn_type **predicted_outputs, nn_type **target_outputs, int batch_size);

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


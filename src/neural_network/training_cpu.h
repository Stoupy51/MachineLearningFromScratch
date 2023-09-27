
#ifndef __NEURAL_TRAINING_CPU_H__
#define __NEURAL_TRAINING_CPU_H__

#include "training_structs.h"

/**
 * @brief Gradients per layer structure.
 * 
 * @param biases_gradients			Pointer to the biases gradients array
 * @param weights_gradients			Pointer to the weights gradients matrix
 * @param weights_gradients_flat	Pointer to the weights gradients flat array
 */
struct gradients_per_layer_t {
	nn_type *biases_gradients;
	nn_type **weights_gradients;
	nn_type *weights_gradients_flat;
};

void shuffleTrainingData(nn_type **inputs, nn_type **target_outputs, int batch_size);
void FeedForwardCPU(NeuralNetwork *network, nn_type **inputs, nn_type **outputs, int batch_size);
void FeedForwardCPUNoInput(NeuralNetwork *network);

int TrainCPU(
	NeuralNetwork *network,
	TrainingData training_data,
	TrainingParameters training_parameters,
	int verbose
);

#endif

/*
	nn_type (*loss_function)(nn_type, nn_type);				// Arbitrary, ex: mean_squared_error, mean_absolute_error, cross_entropy, ...
	nn_type (*loss_function_derivative)(nn_type, nn_type);	// Arbitrary, ex: mean_squared_error_derivative, mean_absolute_error_derivative, cross_entropy_derivative, ...
*/



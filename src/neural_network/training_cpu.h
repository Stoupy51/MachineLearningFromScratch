
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

void FeedForwardCPU(NeuralNetwork *network, nn_type **inputs, nn_type **outputs, int batch_size);
void FeedForwardCPUWithDropout(NeuralNetwork *network, nn_type **inputs, nn_type **outputs, int batch_size, nn_type **dropout_mask, nn_type dropout_scale);
void FeedForwardCPUNoInput(NeuralNetwork *network);

int TrainCPU(
	NeuralNetwork *network,
	TrainingData training_data,
	TrainingParameters training_parameters,
	nn_type *error_per_epoch,
	int verbose
);

#endif



#ifndef __NEURAL_TRAINING_CPU_H__
#define __NEURAL_TRAINING_CPU_H__

#include "training_structs.h"

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


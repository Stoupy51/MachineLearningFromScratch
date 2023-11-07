
#ifndef __NEURAL_TRAINING_GPU_H__
#define __NEURAL_TRAINING_GPU_H__

#include "training_structs.h"

int FeedForwardGPU(NeuralNetwork *network, nn_type **inputs, nn_type **outputs, int batch_size);

int TrainGPU(
	NeuralNetwork *network,
	TrainingData training_data,
	TrainingParameters training_parameters,
	nn_type *error_per_epoch,
	int verbose
);


#endif



#ifndef __NEURAL_TRAINING_GPU_H__
#define __NEURAL_TRAINING_GPU_H__

#include "training_structs.h"

int FeedForwardGPU(NeuralNetwork *network, nn_type **inputs, nn_type **outputs, int batch_size);


#endif


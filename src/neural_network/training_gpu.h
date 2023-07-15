
#ifndef __NEURAL_TRAINING_GPU_H__
#define __NEURAL_TRAINING_GPU_H__

#include "neural_utils.h"

void stopNeuralNetworkGpuOpenCL();
void stopNeuralNetworkGpuBuffersOpenCL();
int NeuralNetworkDfeedForwardGPU(NeuralNetworkD *network, double *input, int read_buffer);
int NeuralNetworkDbackpropagationGPU(NeuralNetworkD *network, double *excepted_output, int read_buffer);
int NeuralNetworkDupdateWeightsAndBiasesGPU(NeuralNetworkD *network, int read_buffer);
int NeuralNetworkDtrainGPU(NeuralNetworkD *network, double *input, double *excepted_output, int read_all_buffers);

#endif



#include "training_gpu.h"
#include "../gpu/gpu_utils.h"
#include "../st_benchmark.h"
#define GPU_TRAINING_BENCHMARK_LEVEL 0

// Variables for gpu efficiency
cl_int gpu_code = 0;
cl_program gpu_program = NULL;
cl_kernel gpu_kernels[] = {NULL, NULL, NULL, NULL};
struct opencl_context_t gpu_oc;
int gpu_initialized = 0, gpu_current_kernel = -1;
cl_mem *activation_values_buffers = NULL;
cl_mem *weights_buffers = NULL;	// Flat versions of the weights of each layer
cl_mem *biases_buffers = NULL;
cl_mem *deltas_buffers = NULL;
NeuralNetworkD *gpu_network = NULL;	// Used to check if the network is the same as the last time
int gpu_network_nb_layers = 0;	// Used to free the buffers

/**
 * @brief This function release the gpu efficiency variables.
 * This function is automatically called by the atexit() function
 * registered in the setupNeuralNetworkGpuOpenCL() function.
 * 
 * @return void
 */
void stopNeuralNetworkGpuOpenCL() {
	if (!gpu_initialized) return;
	clReleaseProgram(gpu_program);
	clReleaseKernels((int)(sizeof(gpu_kernels) / sizeof(cl_kernel)), gpu_kernels);
	clReleaseCommandQueue(gpu_oc.command_queue);
	clReleaseContext(gpu_oc.context);
	gpu_initialized = 0;
}

/**
 * @brief This function release the gpu efficiency variables.
 * This function is automatically called by the atexit() function
 * registered in the setupNeuralNetworkGpuOpenCL() function.
 * 
 * @return void
 */
void stopNeuralNetworkGpuBuffersOpenCL() {
	if (activation_values_buffers == NULL) return;

	// Release the buffers
	for (int i = 0; i < gpu_network_nb_layers; i++) {
		gpu_code = clReleaseMemObject(activation_values_buffers[i]);
		WARNING_HANDLE_INT(gpu_code, "stopNeuralNetworkGpuBuffersOpenCL(): Cannot release buffer 'activation_values_buffers[%d]', reason: %d / %s\n", i, gpu_code, getOpenCLErrorString(gpu_code));
		if (i == 0) continue;
		gpu_code = clReleaseMemObject(weights_buffers[i]);
		WARNING_HANDLE_INT(gpu_code, "stopNeuralNetworkGpuBuffersOpenCL(): Cannot release buffer 'weights_buffers[%d]', reason: %d / %s\n", i, gpu_code, getOpenCLErrorString(gpu_code));
		gpu_code = clReleaseMemObject(biases_buffers[i]);
		WARNING_HANDLE_INT(gpu_code, "stopNeuralNetworkGpuBuffersOpenCL(): Cannot release buffer 'biases_buffers[%d]', reason: %d / %s\n", i, gpu_code, getOpenCLErrorString(gpu_code));
		gpu_code = clReleaseMemObject(deltas_buffers[i]);
		WARNING_HANDLE_INT(gpu_code, "stopNeuralNetworkGpuBuffersOpenCL(): Cannot release buffer 'deltas_buffers[%d]', reason: %d / %s\n", i, gpu_code, getOpenCLErrorString(gpu_code));
	}
	gpu_code = clReleaseMemObject(activation_values_buffers[gpu_network_nb_layers]);
	WARNING_HANDLE_INT(gpu_code, "stopNeuralNetworkGpuBuffersOpenCL(): Cannot release buffer 'activation_values_buffers[%d]', reason: %d / %s\n", gpu_network_nb_layers, gpu_code, getOpenCLErrorString(gpu_code));
	free(activation_values_buffers);
	free(weights_buffers);
	free(biases_buffers);
	free(deltas_buffers);
	activation_values_buffers = NULL;
	weights_buffers = NULL;
	biases_buffers = NULL;
	deltas_buffers = NULL;

	// Reset the network as the last network
	gpu_network = NULL;
}

/**
 * @brief This function setup the gpu efficiency variables.
 * This function is automatically called by the NeuralNetworkDtrainGpuOpenCL() function.
 * 
 * @param network	Pointer to the neural network
 * 
 * @return int		0 if success, -1 otherwise
 */
int setupNeuralNetworkGpuBuffersOpenCL(NeuralNetworkD *network) {
	if (gpu_network == network) return 0;
	if (activation_values_buffers != NULL) stopNeuralNetworkGpuBuffersOpenCL();

	// Create the buffers & copy all the data to the GPU
	activation_values_buffers = (cl_mem*)malloc((network->nb_layers + 1) * sizeof(cl_mem)); // +1 for the output layer
	weights_buffers = (cl_mem*)malloc(network->nb_layers * sizeof(cl_mem));
	biases_buffers = (cl_mem*)malloc(network->nb_layers * sizeof(cl_mem));
	deltas_buffers = (cl_mem*)malloc(network->nb_layers * sizeof(cl_mem));

	// For each layer of the neural network,
	for (int i = 0; i < network->nb_layers; i++) {

		// Create the activation_values buffer
		activation_values_buffers[i] = clCreateBuffer(gpu_oc.context, CL_MEM_READ_WRITE, network->layers[i].nb_neurons * sizeof(double), NULL, &gpu_code);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "setupNeuralNetworkGpuBuffersOpenCL(): Cannot create buffer 'activation_values_buffers[%d]', reason: %d / %s\n", i, gpu_code, getOpenCLErrorString(gpu_code));

		// If it's the input layer, continue
		if (i == 0) continue;

		// Create the weights buffer
		weights_buffers[i] = clCreateBuffer(gpu_oc.context, CL_MEM_READ_WRITE, network->layers[i].nb_neurons * network->layers[i].nb_inputs_per_neuron * sizeof(double), NULL, &gpu_code);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "setupNeuralNetworkGpuBuffersOpenCL(): Cannot create buffer 'weights_buffers[%d]' (size: %d = %d * %d), reason: %d / %s\n", i, network->layers[i].nb_neurons * network->layers[i].nb_inputs_per_neuron, network->layers[i].nb_neurons, network->layers[i].nb_inputs_per_neuron, gpu_code, getOpenCLErrorString(gpu_code));

		// Create the biases buffer
		biases_buffers[i] = clCreateBuffer(gpu_oc.context, CL_MEM_READ_WRITE, network->layers[i].nb_neurons * sizeof(double), NULL, &gpu_code);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "setupNeuralNetworkGpuBuffersOpenCL(): Cannot create buffer 'biases_buffers[%d]', reason: %d / %s\n", i, gpu_code, getOpenCLErrorString(gpu_code));

		// Create the deltas buffer
		deltas_buffers[i] = clCreateBuffer(gpu_oc.context, CL_MEM_READ_WRITE, network->layers[i].nb_neurons * sizeof(double), NULL, &gpu_code);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "setupNeuralNetworkGpuBuffersOpenCL(): Cannot create buffer 'deltas_buffers[%d]', reason: %d / %s\n", i, gpu_code, getOpenCLErrorString(gpu_code));
	}

	// Create the output layer activation_values buffer (for excepted output to avoid creating a new buffer each time)
	activation_values_buffers[network->nb_layers] = clCreateBuffer(gpu_oc.context, CL_MEM_READ_WRITE, network->output_layer->nb_neurons * sizeof(double), NULL, &gpu_code);
	ERROR_HANDLE_INT_RETURN_INT(gpu_code, "setupNeuralNetworkGpuBuffersOpenCL(): Cannot create buffer 'activation_values_buffers[%d]', reason: %d / %s\n", network->nb_layers, gpu_code, getOpenCLErrorString(gpu_code));

	// Write the data to the GPU
	cl_event *write_events_act_val = (cl_event*)malloc(network->nb_layers * sizeof(cl_event));
	cl_event *write_events_others = (cl_event*)malloc((network->nb_layers - 1) * sizeof(cl_event) * 3);
	for (int i = 0; i < network->nb_layers; i++) {
		
		// Write the activation_values buffer
		gpu_code = clEnqueueWriteBuffer(gpu_oc.command_queue, activation_values_buffers[i], CL_FALSE, 0, network->layers[i].nb_neurons * sizeof(double), network->layers[i].activations_values, 0, NULL, &write_events_act_val[i]);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "setupNeuralNetworkGpuBuffersOpenCL(): Cannot write buffer 'activation_values_buffers[%d]', reason: %d / %s\n", i, gpu_code, getOpenCLErrorString(gpu_code));

		// If it's the input layer, continue
		if (i == 0) continue;

		// Write the weights buffer
		gpu_code = clEnqueueWriteBuffer(gpu_oc.command_queue, weights_buffers[i], CL_FALSE, 0, network->layers[i].nb_neurons * network->layers[i].nb_inputs_per_neuron * sizeof(double), network->layers[i].weights_flat, 0, NULL, &write_events_others[(i - 1) * 3]);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "setupNeuralNetworkGpuBuffersOpenCL(): Cannot write buffer 'weights_buffers[%d]', reason: %d / %s\n", i, gpu_code, getOpenCLErrorString(gpu_code));

		// Write the biases buffer
		gpu_code = clEnqueueWriteBuffer(gpu_oc.command_queue, biases_buffers[i], CL_FALSE, 0, network->layers[i].nb_neurons * sizeof(double), network->layers[i].biases, 0, NULL, &write_events_others[(i - 1) * 3 + 1]);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "setupNeuralNetworkGpuBuffersOpenCL(): Cannot write buffer 'biases_buffers[%d]', reason: %d / %s\n", i, gpu_code, getOpenCLErrorString(gpu_code));

		// Write the deltas buffer
		gpu_code = clEnqueueWriteBuffer(gpu_oc.command_queue, deltas_buffers[i], CL_FALSE, 0, network->layers[i].nb_neurons * sizeof(double), network->layers[i].deltas, 0, NULL, &write_events_others[(i - 1) * 3 + 2]);
	}

	// Wait for the write events to finish (skip the first one because it's the input layer)
	gpu_code = clWaitForEvents(network->nb_layers, write_events_act_val);
	ERROR_HANDLE_INT_RETURN_INT(gpu_code, "setupNeuralNetworkGpuBuffersOpenCL(): Cannot wait for 'act_val' events, reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
	gpu_code = clWaitForEvents((network->nb_layers - 1) * 3, write_events_others);
	ERROR_HANDLE_INT_RETURN_INT(gpu_code, "setupNeuralNetworkGpuBuffersOpenCL(): Cannot wait for 'others' events, reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));

	// Free the write events
	for (int i = 0; i < network->nb_layers; i++) {
		gpu_code = clReleaseEvent(write_events_act_val[i]);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "setupNeuralNetworkGpuBuffersOpenCL(): Cannot release 'act_val' event %d, reason: %d / %s\n", i, gpu_code, getOpenCLErrorString(gpu_code));
		if (i == 0) continue;
		gpu_code = clReleaseEvent(write_events_others[(i - 1) * 3]);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "setupNeuralNetworkGpuBuffersOpenCL(): Cannot release 'others' event %d, reason: %d / %s\n", (i - 1) * 3, gpu_code, getOpenCLErrorString(gpu_code));
		gpu_code = clReleaseEvent(write_events_others[(i - 1) * 3 + 1]);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "setupNeuralNetworkGpuBuffersOpenCL(): Cannot release 'others' event %d, reason: %d / %s\n", (i - 1) * 3 + 1, gpu_code, getOpenCLErrorString(gpu_code));
		gpu_code = clReleaseEvent(write_events_others[(i - 1) * 3 + 2]);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "setupNeuralNetworkGpuBuffersOpenCL(): Cannot release 'others' event %d, reason: %d / %s\n", (i - 1) * 3 + 2, gpu_code, getOpenCLErrorString(gpu_code));
	}
	free(write_events_act_val);
	free(write_events_others);

	// Register the exit function
	atexit(stopNeuralNetworkGpuBuffersOpenCL);

	// Set the network as the last network, set the number of layers & return 0
	gpu_network_nb_layers = network->nb_layers;
	gpu_network = network;
	return 0;
}

/**
 * @brief This function setup the gpu efficiency variables.
 * This function is automatically called by the NeuralNetworkDtrainGpuOpenCL() function.
 * 
 * @param network	Pointer to the neural network
 * 
 * @return int		0 if success, -1 otherwise
 */
int setupNeuralNetworkGpuOpenCL(NeuralNetworkD *network) {
	if (gpu_initialized)
		return setupNeuralNetworkGpuBuffersOpenCL(network);

	// Initialize OpenCL
	gpu_oc = setupOpenCL(CL_DEVICE_TYPE_GPU);

	// Create the program
	char path[] = "kernels/neural_network.cl";
	char* kernel_source = readKernelProgram(path);
	ERROR_HANDLE_PTR_RETURN_INT(kernel_source, "setupNeuralNetworkGpuOpenCL(): Cannot read kernel program '%s'\n", path);
	gpu_program = clCreateProgramWithSource(gpu_oc.context, 1, (const char**)&kernel_source, NULL, &gpu_code);
	ERROR_HANDLE_INT_RETURN_INT(gpu_code, "setupNeuralNetworkGpuOpenCL(): Cannot create program, reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));

	// Build the program
	gpu_code = clBuildProgram(gpu_program, 1, &gpu_oc.device_id, NULL, NULL, NULL);
	if (gpu_code != CL_SUCCESS) {
		ERROR_PRINT("setupNeuralNetworkGpuOpenCL(): Cannot build program, reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
		printProgramBuildLog(gpu_program, gpu_oc.device_id, ERROR_LEVEL, "setupNeuralNetworkGpuOpenCL(): ");
		return gpu_code;
	}

	// Free the kernel source code
	free(kernel_source);

	// Create the kernels
	gpu_kernels[0] = clCreateKernel(gpu_program, "feedForwardActivationValuesSigmoid", &gpu_code);
	ERROR_HANDLE_INT_RETURN_INT(gpu_code, "setupNeuralNetworkGpuOpenCL(): Cannot create kernel 'feedForwardActivationValuesSigmoid', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
	gpu_kernels[1] = clCreateKernel(gpu_program, "backpropagationOutputLayerDeltas", &gpu_code);
	ERROR_HANDLE_INT_RETURN_INT(gpu_code, "setupNeuralNetworkGpuOpenCL(): Cannot create kernel 'backpropagationOutputLayerDeltas', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
	gpu_kernels[2] = clCreateKernel(gpu_program, "backpropagationHiddenLayersDeltas", &gpu_code);
	ERROR_HANDLE_INT_RETURN_INT(gpu_code, "setupNeuralNetworkGpuOpenCL(): Cannot create kernel 'backpropagationHiddenLayersDeltas', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
	gpu_kernels[3] = clCreateKernel(gpu_program, "updateWeightsAndBiases", &gpu_code);
	ERROR_HANDLE_INT_RETURN_INT(gpu_code, "setupNeuralNetworkGpuOpenCL(): Cannot create kernel 'updateWeightsAndBiases', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));

	// Register the exit function
	atexit(stopNeuralNetworkGpuOpenCL);

	// Set the initialized flag & return the code of buffers setup
	gpu_initialized = 1;
	return setupNeuralNetworkGpuBuffersOpenCL(network);
}

/**
 * @brief Read all the buffers of the neural network from the GPU to the memory
 * For a neural network using double as type
 * 
 * @param network	Pointer to the neural network
 * 
 * @return void
 */
int NeuralNetworkDReadAllBuffersGPU(NeuralNetworkD *network) {

	// Wait for everything to finish
	gpu_code = clFinish(gpu_oc.command_queue);
	ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDReadAllBuffersGPU(): Cannot wait for command queue, reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));

	// Allocate the events
	// - (nb_layers + 1) for the activation_values_buffers
	// - (nb_layers - 1) for the weights_buffers
	// - (nb_layers - 1) for the biases_buffers
	// - (nb_layers - 1) for the deltas_buffers
	int total_nb_events = network->nb_layers + (network->nb_layers - 1) * 3;
	cl_event *events = (cl_event*)malloc(total_nb_events * sizeof(cl_event));
	ERROR_HANDLE_PTR_RETURN_INT(events, "NeuralNetworkDReadAllBuffersGPU(): Cannot allocate events");
	int event_index = 0;

	// Read the activation_values buffers
	for (int i = 0; i < network->nb_layers; i++) {
		gpu_code = clEnqueueReadBuffer(gpu_oc.command_queue, activation_values_buffers[i], CL_FALSE, 0, network->layers[i].nb_neurons * sizeof(double), network->layers[i].activations_values, 0, NULL, &events[event_index++]);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDReadAllBuffersGPU(): Cannot read buffer 'activation_values_buffers[%d]', reason: %d / %s\n", i, gpu_code, getOpenCLErrorString(gpu_code));
	}

	// Read the weights, biases & deltas buffers
	for (int i = 1; i < network->nb_layers; i++) {
		gpu_code = clEnqueueReadBuffer(gpu_oc.command_queue, weights_buffers[i], CL_FALSE, 0, network->layers[i].nb_neurons * network->layers[i].nb_inputs_per_neuron * sizeof(double), network->layers[i].weights_flat, 0, NULL, &events[event_index++]);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDReadAllBuffersGPU(): Cannot read buffer 'weights_buffers[%d]', reason: %d / %s\n", i, gpu_code, getOpenCLErrorString(gpu_code));
		gpu_code = clEnqueueReadBuffer(gpu_oc.command_queue, biases_buffers[i], CL_FALSE, 0, network->layers[i].nb_neurons * sizeof(double), network->layers[i].biases, 0, NULL, &events[event_index++]);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDReadAllBuffersGPU(): Cannot read buffer 'biases_buffers[%d]', reason: %d / %s\n", i, gpu_code, getOpenCLErrorString(gpu_code));
		gpu_code = clEnqueueReadBuffer(gpu_oc.command_queue, deltas_buffers[i], CL_FALSE, 0, network->layers[i].nb_neurons * sizeof(double), network->layers[i].deltas, 0, NULL, &events[event_index++]);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDReadAllBuffersGPU(): Cannot read buffer 'deltas_buffers[%d]', reason: %d / %s\n", i, gpu_code, getOpenCLErrorString(gpu_code));
	}

	// Wait for the events to finish
	gpu_code = clWaitForEvents(total_nb_events, events);
	ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDReadAllBuffersGPU(): Cannot wait for events, reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));

	// Free the events
	for (int i = 0; i < total_nb_events; i++) {
		gpu_code = clReleaseEvent(events[i]);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDReadAllBuffersGPU(): Cannot release event %d, reason: %d / %s\n", i, gpu_code, getOpenCLErrorString(gpu_code));
	}
	free(events);

	// Return
	return 0;
}

/**
 * @brief Feed forward the neural network with the input array
 * For a neural network using double as type and using GPU
 * 
 * Detailed description of the feed forward algorithm:
 * - Copy the input array into the input layer of the neural network
 * - For each layer of the neural network (except the input layer):
 * 		- For each neuron of the current layer:
 * 			- Calculate the sum of all the weights of the previous layer linked to the current neuron
 * 			- Add the bias of the current neuron
 * 			- Apply the activation function to the weighted sum (often sigmoid)
 * 			- Save the result in the activations_values array of the current layer
 * 
 * Basically, for each layer, for each neuron, the formula is:
 * - activation_value = activation_function( sum( input * weight  ) + bias )
 * - where input is the activation_value of the previous layer
 * 
 * @details i = Index of the selected layer
 * @details j = Index of the selected neuron
 * @details k = Index of the previous layer selected neuron
 * 
 * @param network		Pointer to the neural network
 * @param input			Pointer to the input array (double), must be the same size as the input layer
 * @param read_buffer	Integer, if 1, read the activation_values buffer of the output layer, if 0, don't read it
 * 
 * @return void
 */
int NeuralNetworkDfeedForwardGPU(NeuralNetworkD *network, double *input, int read_buffer) {

	// Setup the private variables if not already done
	gpu_code = setupNeuralNetworkGpuOpenCL(network);
	ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDfeedForwardGPU(): Cannot setup GPU, reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));

	// Set the input layer (copy the input array into the input layer of the neural network)
	gpu_code = clEnqueueWriteBuffer(gpu_oc.command_queue, activation_values_buffers[0], CL_TRUE, 0, network->input_layer->nb_neurons * sizeof(double), input, 0, NULL, NULL);
	ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDfeedForwardGPU(): Cannot write buffer 'activation_values_buffers[0]', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));

	// For each layer of the neural network (except the input layer),
	for (int i = 1; i < network->nb_layers; i++) {

		// Set the kernel arguments
		clSetKernelArg(gpu_kernels[0], 0, sizeof(cl_mem), &activation_values_buffers[i - 1]);		// Previous layer activation values (to read)
		clSetKernelArg(gpu_kernels[0], 1, sizeof(cl_mem), &weights_buffers[i]);						// Current layer weights (to read)
		clSetKernelArg(gpu_kernels[0], 2, sizeof(cl_mem), &biases_buffers[i]);						// Current layer biases (to read)
		clSetKernelArg(gpu_kernels[0], 3, sizeof(cl_mem), &activation_values_buffers[i]);			// Current layer activation values (to write)
		clSetKernelArg(gpu_kernels[0], 4, sizeof(int), &network->layers[i].nb_neurons);				// Current layer nb_neurons
		clSetKernelArg(gpu_kernels[0], 5, sizeof(int), &network->layers[i].nb_inputs_per_neuron);	// Current layer nb_inputs_per_neuron

		// Execute the kernel
		size_t global_dimensions[] = {network->layers[i].nb_neurons, 0, 0};
		clEnqueueNDRangeKernel(gpu_oc.command_queue, gpu_kernels[0], 1, NULL, global_dimensions, NULL, 0, NULL, NULL);
	}

	// Read the output layer activation_values buffer if needed
	if (read_buffer) {
		gpu_code = clEnqueueReadBuffer(gpu_oc.command_queue, activation_values_buffers[network->nb_layers - 1], CL_TRUE, 0, network->output_layer->nb_neurons * sizeof(double), network->output_layer->activations_values, 0, NULL, NULL);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDfeedForwardGPU(): Cannot read buffer 'activation_values_buffers[%d]', reason: %d / %s\n", network->nb_layers - 1, gpu_code, getOpenCLErrorString(gpu_code));
	}

	// Return
	return 0;
}

/**
 * @brief Backpropagate the neural network with the excepted output array
 * For a neural network using double as type and using GPU
 * 
 * Detailed description of the backpropagation algorithm:
 * - For each neuron of the output layer:
 * 		- Calculate the error of the neuron (excepted_output - activation_value)
 * 		- Calculate the derivative of the activation function of the neuron (activation_value * (1 - activation_value))
 * 		- Calculate the delta of the neuron (error * derivative)
 * 		- Save the delta of the neuron in the deltas array of the output layer
 * - For each layer of the neural network (except the output layer and the input layer) (order last to first):
 * 		- For each neuron of the current layer:
 * 			- Calculate the error of the neuron (sum( weight * delta of the next layer ))
 * 			- Calculate the derivative of the activation function of the neuron (activation_value * (1 - activation_value))
 * 			- Calculate the delta of the neuron (error * derivative)
 * 			- Save the delta of the neuron in the deltas array of the current layer
 * 
 * Basically, for each layer, for each neuron, the formula is:
 * - delta = error * derivative
 * - where error = sum( weight * delta of the next layer )
 * - where derivative = activation_value * (1 - activation_value)
 * 
 * @details i = Index of the selected layer (order last to first)
 * @details j = Index of the selected neuron
 * @details k = Index of the next layer selected neuron
 * 
 * @param network			Pointer to the neural network
 * @param excepted_output	Pointer to the excepted output array (double), must be the same size as the output layer
 * @param read_buffer		Integer, if 1, read the all the deltas buffers, if 0, don't read them
 * 
 * @return void
 */
int NeuralNetworkDbackpropagationGPU(NeuralNetworkD *network, double *excepted_output, int read_buffer) {

	// Setup the private variables if not already done
	gpu_code = setupNeuralNetworkGpuOpenCL(network);
	ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDbackpropagationGPU(): Cannot setup GPU, reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));

	// Create the buffers & copy all the data to the GPU
	cl_mem excepted_output_buffer = clCreateBuffer(gpu_oc.context, CL_MEM_READ_ONLY, network->output_layer->nb_neurons * sizeof(double), NULL, &gpu_code);
	ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDbackpropagationGPU(): Cannot create buffer 'excepted_output_buffer', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
	gpu_code = clEnqueueWriteBuffer(gpu_oc.command_queue, excepted_output_buffer, CL_TRUE, 0, network->output_layer->nb_neurons * sizeof(double), excepted_output, 0, NULL, NULL);
	ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDbackpropagationGPU(): Cannot write buffer 'excepted_output_buffer', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));

	// For each neuron of the output layer, calculate the delta of the neuron (excepted_output - activation_value) * derivative
	int output_layer_size = network->output_layer->nb_neurons;
	gpu_code = clSetKernelArg(gpu_kernels[1], 0, sizeof(cl_mem), &excepted_output_buffer);
	ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDbackpropagationGPU(): Cannot set kernel argument 0 of kernel 'backpropagationOutputLayerDeltas', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
	gpu_code = clSetKernelArg(gpu_kernels[1], 1, sizeof(cl_mem), &activation_values_buffers[network->nb_layers - 1]);
	ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDbackpropagationGPU(): Cannot set kernel argument 1 of kernel 'backpropagationOutputLayerDeltas', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
	gpu_code = clSetKernelArg(gpu_kernels[1], 2, sizeof(cl_mem), &deltas_buffers[network->nb_layers - 1]);
	ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDbackpropagationGPU(): Cannot set kernel argument 2 of kernel 'backpropagationOutputLayerDeltas', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
	gpu_code = clSetKernelArg(gpu_kernels[1], 3, sizeof(int), &output_layer_size);
	ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDbackpropagationGPU(): Cannot set kernel argument 3 of kernel 'backpropagationOutputLayerDeltas', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
	size_t global_dimensions[] = { network->output_layer->nb_neurons, 0, 0 };
	gpu_code = clEnqueueNDRangeKernel(gpu_oc.command_queue, gpu_kernels[1], 1, NULL, global_dimensions, NULL, 0, NULL, NULL);
	ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDbackpropagationGPU(): Cannot enqueue kernel 'backpropagationOutputLayerDeltas', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));

	// For each layer of the neural network (except the output layer and the input layer) (order last to first),
	for (int i = network->nb_layers - 2; i > 0; i--) {

		// For each neuron of the current layer,
		int current_layer_size = network->layers[i].nb_neurons;
		int next_layer_size = network->layers[i + 1].nb_neurons;
		gpu_code = clSetKernelArg(gpu_kernels[2], 0, sizeof(cl_mem), &weights_buffers[i + 1]);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDbackpropagationGPU(): Cannot set kernel argument 0 of kernel 'backpropagationHiddenLayersDeltas', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
		gpu_code = clSetKernelArg(gpu_kernels[2], 1, sizeof(cl_mem), &deltas_buffers[i + 1]);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDbackpropagationGPU(): Cannot set kernel argument 1 of kernel 'backpropagationHiddenLayersDeltas', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
		gpu_code = clSetKernelArg(gpu_kernels[2], 2, sizeof(cl_mem), &activation_values_buffers[i]);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDbackpropagationGPU(): Cannot set kernel argument 2 of kernel 'backpropagationHiddenLayersDeltas', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
		gpu_code = clSetKernelArg(gpu_kernels[2], 3, sizeof(cl_mem), &deltas_buffers[i]);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDbackpropagationGPU(): Cannot set kernel argument 3 of kernel 'backpropagationHiddenLayersDeltas', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
		gpu_code = clSetKernelArg(gpu_kernels[2], 4, sizeof(int), &current_layer_size);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDbackpropagationGPU(): Cannot set kernel argument 4 of kernel 'backpropagationHiddenLayersDeltas', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
		gpu_code = clSetKernelArg(gpu_kernels[2], 5, sizeof(int), &next_layer_size);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDbackpropagationGPU(): Cannot set kernel argument 5 of kernel 'backpropagationHiddenLayersDeltas', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
		size_t global_work_size[] = {current_layer_size};
		gpu_code = clEnqueueNDRangeKernel(gpu_oc.command_queue, gpu_kernels[2], 1, NULL, global_work_size, NULL, 0, NULL, NULL);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDbackpropagationGPU(): Cannot enqueue kernel 'backpropagationHiddenLayersDeltas', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
	}

	// Copy the deltas back to the CPU if needed
	if (read_buffer) {
		gpu_code = clFinish(gpu_oc.command_queue);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDbackpropagationGPU(): Cannot finish command queue, reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
		cl_event *read_events = (cl_event*)malloc((network->nb_layers - 1) * sizeof(cl_event));
		for (int i = 1; i < network->nb_layers; i++) {
			gpu_code = clEnqueueReadBuffer(gpu_oc.command_queue, deltas_buffers[i], CL_FALSE, 0, network->layers[i].nb_neurons * sizeof(double), network->layers[i].deltas, 0, NULL, &read_events[i - 1]);
			ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDbackpropagationGPU(): Cannot read buffer 'deltas_buffers[%d]', reason: %d / %s\n", i, gpu_code, getOpenCLErrorString(gpu_code));
		}
		gpu_code = clWaitForEvents(network->nb_layers - 1, read_events);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDbackpropagationGPU(): Cannot wait for events 'read_events[1:%d]', reason: %d / %s\n", network->nb_layers - 2, gpu_code, getOpenCLErrorString(gpu_code));
		for (int i = 1; i < network->nb_layers; i++) {
			gpu_code = clReleaseEvent(read_events[i - 1]);
			ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDbackpropagationGPU(): Cannot release event 'read_events[%d]', reason: %d / %s\n", i, gpu_code, getOpenCLErrorString(gpu_code));
		}
		free(read_events);
	}

	// Return
	return 0;
}

/**
 * @brief Update/Nudge the weights and biases of the neural network
 * For a neural network using double as type and using GPU
 * 
 * Detailed description of the update weights and biases algorithm:
 * - For each layer of the neural network (except the input layer):
 * 		- For each neuron of the current layer:
 * 			- For each weight of the current neuron:
 * 				- Update the weight (weight + (learning_rate * delta of the current neuron * activation_value of the previous layer))
 * 			- Update the bias (bias + (learning_rate * delta of the current neuron))
 * 
 * Basically, for each layer, for each neuron, the formula is:
 * - bias = bias + (learning_rate * delta of the current neuron)
 * - for each weight of the current neuron:
 *   - weight = weight + (learning_rate * delta of the current neuron * activation_value of the previous layer)
 * 
 * @details i = Index of the selected layer (except the input layer)
 * @details j = Index of the selected neuron
 * @details k = Index of the previous layer selected neuron
 * 
 * @param network		Pointer to the neural network
 * @param read_buffer	Integer, if 1, read the all the weights and biases buffers, if 0, don't read them
 * 
 * @return void
 */
int NeuralNetworkDupdateWeightsAndBiasesGPU(NeuralNetworkD *network, int read_buffer) {

	// Setup the private variables if not already done
	gpu_code = setupNeuralNetworkGpuOpenCL(network);
	ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDupdateWeightsAndBiasesGPU(): Cannot setup GPU, reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));

	// For each layer of the neural network (except the input layer),
	for (int i = 1; i < network->nb_layers; i++) {

		// For each neuron of the current layer,
		int current_layer_size = network->layers[i].nb_neurons;
		int previous_layer_size = network->layers[i - 1].nb_neurons;
		double learning_rate = network->learning_rate;
		gpu_code = clSetKernelArg(gpu_kernels[3], 0, sizeof(cl_mem), &activation_values_buffers[i - 1]);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDupdateWeightsAndBiasesGPU(): Cannot set kernel argument 0 of kernel 'updateWeightsAndBiases', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
		gpu_code = clSetKernelArg(gpu_kernels[3], 1, sizeof(cl_mem), &deltas_buffers[i]);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDupdateWeightsAndBiasesGPU(): Cannot set kernel argument 1 of kernel 'updateWeightsAndBiases', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
		gpu_code = clSetKernelArg(gpu_kernels[3], 2, sizeof(cl_mem), &weights_buffers[i]);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDupdateWeightsAndBiasesGPU(): Cannot set kernel argument 2 of kernel 'updateWeightsAndBiases', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
		gpu_code = clSetKernelArg(gpu_kernels[3], 3, sizeof(cl_mem), &biases_buffers[i]);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDupdateWeightsAndBiasesGPU(): Cannot set kernel argument 3 of kernel 'updateWeightsAndBiases', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
		gpu_code = clSetKernelArg(gpu_kernels[3], 4, sizeof(int), &current_layer_size);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDupdateWeightsAndBiasesGPU(): Cannot set kernel argument 4 of kernel 'updateWeightsAndBiases', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
		gpu_code = clSetKernelArg(gpu_kernels[3], 5, sizeof(int), &previous_layer_size);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDupdateWeightsAndBiasesGPU(): Cannot set kernel argument 5 of kernel 'updateWeightsAndBiases', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
		gpu_code = clSetKernelArg(gpu_kernels[3], 6, sizeof(double), &learning_rate);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDupdateWeightsAndBiasesGPU(): Cannot set kernel argument 6 of kernel 'updateWeightsAndBiases', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
		size_t global_work_size[] = {current_layer_size};
		gpu_code = clEnqueueNDRangeKernel(gpu_oc.command_queue, gpu_kernels[3], 1, NULL, global_work_size, NULL, 0, NULL, NULL);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDupdateWeightsAndBiasesGPU(): Cannot enqueue kernel 'updateWeightsAndBiases', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
	}

	// Copy the data from the GPU to the CPU (except the input layer): weights and biases buffers if needed
	if (read_buffer) {
		gpu_code = clFinish(gpu_oc.command_queue);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDupdateWeightsAndBiasesGPU(): Cannot finish command queue, reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
		cl_event *write_events = (cl_event*)malloc((network->nb_layers - 1) * sizeof(cl_event) * 2);
		for (int i = 1; i < network->nb_layers; i++) {
			gpu_code = clEnqueueReadBuffer(gpu_oc.command_queue, weights_buffers[i], CL_FALSE, 0, network->layers[i].nb_neurons * network->layers[i].nb_inputs_per_neuron * sizeof(double), network->layers[i].weights_flat, 0, NULL, &write_events[(i - 1) * 2]);
			ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDupdateWeightsAndBiasesGPU(): Cannot read buffer 'weights_buffers[%d]', reason: %d / %s\n", i, gpu_code, getOpenCLErrorString(gpu_code));
			gpu_code = clEnqueueReadBuffer(gpu_oc.command_queue, biases_buffers[i], CL_FALSE, 0, network->layers[i].nb_neurons * sizeof(double), network->layers[i].biases, 0, NULL, &write_events[(i - 1) * 2 + 1]);
			ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDupdateWeightsAndBiasesGPU(): Cannot read buffer 'biases_buffers[%d]', reason: %d / %s\n", i, gpu_code, getOpenCLErrorString(gpu_code));
		}
		gpu_code = clWaitForEvents((network->nb_layers - 1) * 2, write_events);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDupdateWeightsAndBiasesGPU(): Cannot wait for events 'write_events[1:%d]', reason: %d / %s\n", (network->nb_layers - 1) * 2 - 1, gpu_code, getOpenCLErrorString(gpu_code));
		for (int i = 1; i < network->nb_layers; i++) {
			gpu_code = clReleaseEvent(write_events[(i - 1) * 2]);
			ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDupdateWeightsAndBiasesGPU(): Cannot release event 'write_events[%d]', reason: %d / %s\n", (i - 1) * 2, gpu_code, getOpenCLErrorString(gpu_code));
			gpu_code = clReleaseEvent(write_events[(i - 1) * 2 + 1]);
			ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDupdateWeightsAndBiasesGPU(): Cannot release event 'write_events[%d]', reason: %d / %s\n", (i - 1) * 2 + 1, gpu_code, getOpenCLErrorString(gpu_code));
		}
		free(write_events);
	}

	// Return
	return 0;
}

/**
 * @brief Train the neural network with the input array and the excepted output array step by step
 * For a neural network using double as type and using GPU
 * 
 * @param network			Pointer to the neural network
 * @param input				Pointer to the input array (double), must be the same size as the input layer
 * @param excepted_output	Pointer to the excepted output array (double), must be the same size as the output layer
 * @param read_all_buffers	Integer, if 1, read all the buffers, if 0, don't read them
 * 
 * @return void
 */
int NeuralNetworkDtrainStepByStepGPU(NeuralNetworkD *network, double *input, double *excepted_output, int read_all_buffers) {

	// Feed forward
	gpu_code = NeuralNetworkDfeedForwardGPU(network, input, 0);
	ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDtrainGPU(): Cannot feed forward, reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));

	// Backpropagation
	gpu_code = NeuralNetworkDbackpropagationGPU(network, excepted_output, 0);
	ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDtrainGPU(): Cannot backpropagate, reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));

	// Update weights and biases
	gpu_code = NeuralNetworkDupdateWeightsAndBiasesGPU(network, 0);
	ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDtrainGPU(): Cannot update weights and biases, reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));

	// Read all the buffers if needed
	if (read_all_buffers) {
		gpu_code = NeuralNetworkDReadAllBuffersGPU(network);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDtrainGPU(): Cannot read all buffers, reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
	}

	// Return
	return 0;
}

/**
 * @brief Train the neural network all at once with the input array and the excepted output array
 * For a neural network using double as type and using GPU
 * Equivalent to call NeuralNetworkDfeedForwardGPU(), NeuralNetworkDbackpropagationGPU() and NeuralNetworkDupdateWeightsAndBiasesGPU() in a row
 * 
 * @param network			Pointer to the neural network
 * @param input				Pointer to the input array (double), must be the same size as the input layer
 * @param excepted_output	Pointer to the excepted output array (double), must be the same size as the output layer
 * @param read_all_buffers	Integer, if 1, read all the buffers, if 0, don't read them
 * 
 * @return void
 */
int NeuralNetworkDtrainGPU(NeuralNetworkD *network, double *input, double *excepted_output, int read_all_buffers) {

	// Setup the private variables if not already done
	gpu_code = setupNeuralNetworkGpuOpenCL(network);
	ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDtrainGPU(): Cannot setup GPU, reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));

	// Allocate memory for cl_event objects, so we need:
	// - x(1) cl_event for the excepted output buffer
	// - x(1) cl_event for the input layer
	// - x(nb_layers - 1) cl_event for the feed forward (1 per layer)
	// - x(nb_layers - 1) cl_event for the backpropagation (1 per layer)
	// - x(nb_layers - 1) cl_event for the update weights and biases (1 per layer)
	int total_events = 2 + (network->nb_layers - 1) * 3;
	cl_event *events = malloc(total_events * sizeof(cl_event));
	ERROR_HANDLE_PTR_RETURN_INT(events, "NeuralNetworkDtrainGPU(): Cannot allocate memory for 'events'\n");

	// Write in the buffer for the excepted output array for the backpropagation later
	int event_index = 0;
	gpu_code = clEnqueueWriteBuffer(gpu_oc.command_queue, activation_values_buffers[network->nb_layers], CL_FALSE, 0, network->output_layer->nb_neurons * sizeof(double), excepted_output, 0, NULL, &events[event_index]);
	ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDtrainGPU(): Cannot write buffer 'activation_values_buffers[%d]', reason: %d / %s\n", network->nb_layers, gpu_code, getOpenCLErrorString(gpu_code));

	///// Feed forward part /////
	// Set the input layer (copy the input array into the input layer of the neural network)
	event_index++;
	gpu_code = clEnqueueWriteBuffer(gpu_oc.command_queue, activation_values_buffers[0], CL_FALSE, 0, network->input_layer->nb_neurons * sizeof(double), input, 0, NULL, &events[event_index]);
	ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDtrainGPU(): Cannot write buffer 'activation_values_buffers[0]', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));

	// For each layer of the neural network (except the input layer),
	for (int i = 1; i < network->nb_layers; i++) {

		// Increment the event index
		event_index++;

		// For each neuron of the current layer, calculate the activation_value of the neuron
		gpu_code = clSetKernelArg(gpu_kernels[0], 0, sizeof(cl_mem), &activation_values_buffers[i - 1]);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDtrainGPU(): Cannot set kernel argument 0 of kernel 'feedForwardActivationValuesSigmoid', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
		gpu_code = clSetKernelArg(gpu_kernels[0], 1, sizeof(cl_mem), &weights_buffers[i]);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDtrainGPU(): Cannot set kernel argument 1 of kernel 'feedForwardActivationValuesSigmoid', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
		gpu_code = clSetKernelArg(gpu_kernels[0], 2, sizeof(cl_mem), &biases_buffers[i]);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDtrainGPU(): Cannot set kernel argument 2 of kernel 'feedForwardActivationValuesSigmoid', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
		gpu_code = clSetKernelArg(gpu_kernels[0], 3, sizeof(cl_mem), &activation_values_buffers[i]);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDtrainGPU(): Cannot set kernel argument 3 of kernel 'feedForwardActivationValuesSigmoid', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
		gpu_code = clSetKernelArg(gpu_kernels[0], 4, sizeof(int), &network->layers[i].nb_neurons);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDtrainGPU(): Cannot set kernel argument 4 of kernel 'feedForwardActivationValuesSigmoid', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
		gpu_code = clSetKernelArg(gpu_kernels[0], 5, sizeof(int), &network->layers[i].nb_inputs_per_neuron);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDtrainGPU(): Cannot set kernel argument 5 of kernel 'feedForwardActivationValuesSigmoid', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
		size_t global_work_size[] = {network->layers[i].nb_neurons};
		gpu_code = clEnqueueNDRangeKernel(gpu_oc.command_queue, gpu_kernels[0], 1, NULL, global_work_size, NULL, 1, &events[event_index - 1], &events[event_index]);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDtrainGPU(): Cannot enqueue kernel 'feedForwardActivationValuesSigmoid', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
	}

	///// Backpropagation part /////
	// Prepare wait list for the last event & output buffer
	cl_event wait_list[2] = {events[event_index], events[0]};

	// For each neuron of the output layer, calculate the delta of the neuron (excepted_output - activation_value) * derivative
	event_index++;
	gpu_code = clSetKernelArg(gpu_kernels[1], 0, sizeof(cl_mem), &activation_values_buffers[network->nb_layers]);
	ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDtrainGPU(): Cannot set kernel argument 0 of kernel 'backpropagationOutputLayerDeltas', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
	gpu_code = clSetKernelArg(gpu_kernels[1], 1, sizeof(cl_mem), &activation_values_buffers[network->nb_layers - 1]);
	ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDtrainGPU(): Cannot set kernel argument 1 of kernel 'backpropagationOutputLayerDeltas', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
	gpu_code = clSetKernelArg(gpu_kernels[1], 2, sizeof(cl_mem), &deltas_buffers[network->nb_layers - 1]);
	ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDtrainGPU(): Cannot set kernel argument 2 of kernel 'backpropagationOutputLayerDeltas', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
	gpu_code = clSetKernelArg(gpu_kernels[1], 3, sizeof(int), &network->output_layer->nb_neurons);
	ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDtrainGPU(): Cannot set kernel argument 3 of kernel 'backpropagationOutputLayerDeltas', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
	size_t global_work_size[] = {network->output_layer->nb_neurons};
	gpu_code = clEnqueueNDRangeKernel(gpu_oc.command_queue, gpu_kernels[1], 1, NULL, global_work_size, NULL, 2, wait_list, &events[event_index]);
	ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDtrainGPU(): Cannot enqueue kernel 'backpropagationOutputLayerDeltas', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));

	// For each layer of the neural network (except the output layer and the input layer) (order last to first),
	for (int i = network->nb_layers - 2; i > 0; i--) {

		// Increment the event index
		event_index++;

		// For each neuron of the current layer,
		gpu_code = clSetKernelArg(gpu_kernels[2], 0, sizeof(cl_mem), &weights_buffers[i + 1]);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDtrainGPU(): Cannot set kernel argument 0 of kernel 'backpropagationHiddenLayersDeltas', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
		gpu_code = clSetKernelArg(gpu_kernels[2], 1, sizeof(cl_mem), &deltas_buffers[i + 1]);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDtrainGPU(): Cannot set kernel argument 1 of kernel 'backpropagationHiddenLayersDeltas', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
		gpu_code = clSetKernelArg(gpu_kernels[2], 2, sizeof(cl_mem), &activation_values_buffers[i]);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDtrainGPU(): Cannot set kernel argument 2 of kernel 'backpropagationHiddenLayersDeltas', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
		gpu_code = clSetKernelArg(gpu_kernels[2], 3, sizeof(cl_mem), &deltas_buffers[i]);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDtrainGPU(): Cannot set kernel argument 3 of kernel 'backpropagationHiddenLayersDeltas', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
		gpu_code = clSetKernelArg(gpu_kernels[2], 4, sizeof(int), &network->layers[i].nb_neurons);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDtrainGPU(): Cannot set kernel argument 4 of kernel 'backpropagationHiddenLayersDeltas', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
		gpu_code = clSetKernelArg(gpu_kernels[2], 5, sizeof(int), &network->layers[i + 1].nb_neurons);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDtrainGPU(): Cannot set kernel argument 5 of kernel 'backpropagationHiddenLayersDeltas', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
		size_t global_work_size[] = {network->layers[i].nb_neurons};
		gpu_code = clEnqueueNDRangeKernel(gpu_oc.command_queue, gpu_kernels[2], 1, NULL, global_work_size, NULL, 1, &events[event_index - 1], &events[event_index]);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDtrainGPU(): Cannot enqueue kernel 'backpropagationHiddenLayersDeltas', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
	}

	///// Update weights and biases part /////
	// For each layer of the neural network (except the input layer) (order last to first),
	for (int i = network->nb_layers - 1; i > 0; i--) {

		// Increment the event index
		event_index++;

		// For each neuron of the current layer,
		gpu_code = clSetKernelArg(gpu_kernels[3], 0, sizeof(cl_mem), &activation_values_buffers[i - 1]);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDtrainGPU(): Cannot set kernel argument 0 of kernel 'updateWeightsAndBiases', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
		gpu_code = clSetKernelArg(gpu_kernels[3], 1, sizeof(cl_mem), &deltas_buffers[i]);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDtrainGPU(): Cannot set kernel argument 1 of kernel 'updateWeightsAndBiases', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
		gpu_code = clSetKernelArg(gpu_kernels[3], 2, sizeof(cl_mem), &weights_buffers[i]);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDtrainGPU(): Cannot set kernel argument 2 of kernel 'updateWeightsAndBiases', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
		gpu_code = clSetKernelArg(gpu_kernels[3], 3, sizeof(cl_mem), &biases_buffers[i]);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDtrainGPU(): Cannot set kernel argument 3 of kernel 'updateWeightsAndBiases', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
		gpu_code = clSetKernelArg(gpu_kernels[3], 4, sizeof(int), &network->layers[i].nb_neurons);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDtrainGPU(): Cannot set kernel argument 4 of kernel 'updateWeightsAndBiases', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
		gpu_code = clSetKernelArg(gpu_kernels[3], 5, sizeof(int), &network->layers[i].nb_inputs_per_neuron);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDtrainGPU(): Cannot set kernel argument 5 of kernel 'updateWeightsAndBiases', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
		gpu_code = clSetKernelArg(gpu_kernels[3], 6, sizeof(double), &network->learning_rate);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDtrainGPU(): Cannot set kernel argument 6 of kernel 'updateWeightsAndBiases', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
		size_t global_work_size[] = {network->layers[i].nb_neurons};
		int backpropa_index = event_index - network->nb_layers;
		gpu_code = clEnqueueNDRangeKernel(gpu_oc.command_queue, gpu_kernels[3], 1, NULL, global_work_size, NULL, 1, &events[backpropa_index], &events[event_index]);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDtrainGPU(): Cannot enqueue kernel 'updateWeightsAndBiases', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
	}

	// Wait for the last events to finish
	int index_start = event_index - network->nb_layers + 1;
	gpu_code = clWaitForEvents(network->nb_layers - 1, &events[index_start]);
	ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDtrainGPU(): Cannot wait for events 'events[%d:%d]', reason: %d / %s\n", index_start, event_index, gpu_code, getOpenCLErrorString(gpu_code));

	// Read all the buffers if needed
	if (read_all_buffers) {
		gpu_code = NeuralNetworkDReadAllBuffersGPU(network);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDtrainGPU(): Cannot read all buffers, reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
	}

	// Release the events
	for (int i = 0; i < total_events; i++) {
		gpu_code = clReleaseEvent(events[i]);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDtrainGPU(): Cannot release event 'events[%d]', reason: %d / %s\n", i, gpu_code, getOpenCLErrorString(gpu_code));
	}
	free(events);

	// Return
	return 0;
}

/**
 * @brief Train the neural network with an image from the image list
 * For a neural network using double as type and using GPU
 * 
 * @param network			Pointer to the neural network
 * @param events			Pointer to the events array
 * @param event_index		Pointer to the event index
 * @param current_elt		Pointer to the current image list element
 * @param image_index		Integer, index of the image in the image list
 * 
 * @return void
 */
int NeuralNetworkDtrainFromImageListPVImgGPU(NeuralNetworkD *network, cl_event *events, int *event_index, image_t image, int image_index, double **images_datas, cl_event excepted_output_buffer_event) {

	// Write the image data into the input layer with the image ratio
	double input_output_ratio = (double)(image.width * image.height) / (double)(network->output_layer->nb_neurons);
	int image_size = network->input_layer->nb_neurons;
	images_datas[image_index] = image_to_double_array(image, image_size, 1);
	images_datas[image_index][0] = input_output_ratio;
	if (*event_index == 0)
		gpu_code = clEnqueueWriteBuffer(gpu_oc.command_queue, activation_values_buffers[0], CL_FALSE, 0, image_size * sizeof(double), images_datas[image_index], 0, NULL, &events[*event_index]);
	else
		gpu_code = clEnqueueWriteBuffer(gpu_oc.command_queue, activation_values_buffers[0], CL_FALSE, 0, image_size * sizeof(double), images_datas[image_index], 1, &events[*event_index - 1], &events[*event_index]);
	ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDtrainFromImageListGPU(): Cannot write buffer 'activation_values_buffers[0]', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));

	///// Feed forward
	// For each layer of the neural network (except the input layer),
	for (int i = 1; i < network->nb_layers; i++) {

		// Next event index
		(*event_index)++;

		// For each neuron of the current layer, calculate the activation_value of the neuron
		gpu_code = clSetKernelArg(gpu_kernels[0], 0, sizeof(cl_mem), &activation_values_buffers[i - 1]);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDtrainFromImageListGPU(): Cannot set kernel argument 0 of kernel 'feedForwardActivationValuesSigmoid', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
		gpu_code = clSetKernelArg(gpu_kernels[0], 1, sizeof(cl_mem), &weights_buffers[i]);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDtrainFromImageListGPU(): Cannot set kernel argument 1 of kernel 'feedForwardActivationValuesSigmoid', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
		gpu_code = clSetKernelArg(gpu_kernels[0], 2, sizeof(cl_mem), &biases_buffers[i]);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDtrainFromImageListGPU(): Cannot set kernel argument 2 of kernel 'feedForwardActivationValuesSigmoid', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
		gpu_code = clSetKernelArg(gpu_kernels[0], 3, sizeof(cl_mem), &activation_values_buffers[i]);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDtrainFromImageListGPU(): Cannot set kernel argument 3 of kernel 'feedForwardActivationValuesSigmoid', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
		gpu_code = clSetKernelArg(gpu_kernels[0], 4, sizeof(int), &network->layers[i].nb_neurons);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDtrainFromImageListGPU(): Cannot set kernel argument 4 of kernel 'feedForwardActivationValuesSigmoid', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
		gpu_code = clSetKernelArg(gpu_kernels[0], 5, sizeof(int), &network->layers[i].nb_inputs_per_neuron);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDtrainFromImageListGPU(): Cannot set kernel argument 5 of kernel 'feedForwardActivationValuesSigmoid', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
		size_t global_work_size[] = {network->layers[i].nb_neurons};
		gpu_code = clEnqueueNDRangeKernel(gpu_oc.command_queue, gpu_kernels[0], 1, NULL, global_work_size, NULL, 1, &events[*event_index - 1], &events[*event_index]);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDtrainFromImageListGPU(): Cannot enqueue kernel 'feedForwardActivationValuesSigmoid', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
	}

	///// Backpropagation
	// Prepare wait list for the last feed forward & output buffer
	cl_event wait_list[2] = {events[*event_index], excepted_output_buffer_event};
	(*event_index)++;

	// For each neuron of the output layer, calculate the delta of the neuron (excepted_output - activation_value) * derivative
	gpu_code = clSetKernelArg(gpu_kernels[1], 0, sizeof(cl_mem), &activation_values_buffers[network->nb_layers - 1]);
	ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDtrainFromImageListGPU(): Cannot set kernel argument 0 of kernel 'backpropagationOutputLayerDeltas', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
	gpu_code = clSetKernelArg(gpu_kernels[1], 1, sizeof(cl_mem), &activation_values_buffers[network->nb_layers - 1]);
	ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDtrainFromImageListGPU(): Cannot set kernel argument 1 of kernel 'backpropagationOutputLayerDeltas', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
	gpu_code = clSetKernelArg(gpu_kernels[1], 2, sizeof(cl_mem), &deltas_buffers[network->nb_layers - 1]);
	ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDtrainFromImageListGPU(): Cannot set kernel argument 2 of kernel 'backpropagationOutputLayerDeltas', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
	gpu_code = clSetKernelArg(gpu_kernels[1], 3, sizeof(int), &network->output_layer->nb_neurons);
	ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDtrainFromImageListGPU(): Cannot set kernel argument 3 of kernel 'backpropagationOutputLayerDeltas', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
	size_t global_work_size[] = {network->output_layer->nb_neurons};
	gpu_code = clEnqueueNDRangeKernel(gpu_oc.command_queue, gpu_kernels[1], 1, NULL, global_work_size, NULL, 2, wait_list, &events[*event_index]);
	ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDtrainFromImageListGPU(): Cannot enqueue kernel 'backpropagationOutputLayerDeltas', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));

	// For each layer of the neural network (except the output layer and the input layer) (order last to first),
	for (int i = network->nb_layers - 2; i > 0; i--) {

		// Next event index
		(*event_index)++;

		// For each neuron of the current layer,
		gpu_code = clSetKernelArg(gpu_kernels[2], 0, sizeof(cl_mem), &weights_buffers[i + 1]);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDtrainFromImageListGPU(): Cannot set kernel argument 0 of kernel 'backpropagationHiddenLayersDeltas', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
		gpu_code = clSetKernelArg(gpu_kernels[2], 1, sizeof(cl_mem), &deltas_buffers[i + 1]);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDtrainFromImageListGPU(): Cannot set kernel argument 1 of kernel 'backpropagationHiddenLayersDeltas', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
		gpu_code = clSetKernelArg(gpu_kernels[2], 2, sizeof(cl_mem), &activation_values_buffers[i]);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDtrainFromImageListGPU(): Cannot set kernel argument 2 of kernel 'backpropagationHiddenLayersDeltas', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
		gpu_code = clSetKernelArg(gpu_kernels[2], 3, sizeof(cl_mem), &deltas_buffers[i]);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDtrainFromImageListGPU(): Cannot set kernel argument 3 of kernel 'backpropagationHiddenLayersDeltas', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
		gpu_code = clSetKernelArg(gpu_kernels[2], 4, sizeof(int), &network->layers[i].nb_neurons);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDtrainFromImageListGPU(): Cannot set kernel argument 4 of kernel 'backpropagationHiddenLayersDeltas', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
		gpu_code = clSetKernelArg(gpu_kernels[2], 5, sizeof(int), &network->layers[i + 1].nb_neurons);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDtrainFromImageListGPU(): Cannot set kernel argument 5 of kernel 'backpropagationHiddenLayersDeltas', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
		size_t global_work_size[] = {network->layers[i].nb_neurons};
		gpu_code = clEnqueueNDRangeKernel(gpu_oc.command_queue, gpu_kernels[2], 1, NULL, global_work_size, NULL, 1, &events[*event_index - 1], &events[*event_index]);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDtrainFromImageListGPU(): Cannot enqueue kernel 'backpropagationHiddenLayersDeltas', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
	}

	///// Update weights and biases
	// For each layer of the neural network (except the input layer) (order last to first),
	for (int i = network->nb_layers - 1; i > 0; i--) {

		// Next event index
		(*event_index)++;

		// For each neuron of the current layer,
		gpu_code = clSetKernelArg(gpu_kernels[3], 0, sizeof(cl_mem), &activation_values_buffers[i - 1]);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDtrainFromImageListGPU(): Cannot set kernel argument 0 of kernel 'updateWeightsAndBiases', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
		gpu_code = clSetKernelArg(gpu_kernels[3], 1, sizeof(cl_mem), &deltas_buffers[i]);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDtrainFromImageListGPU(): Cannot set kernel argument 1 of kernel 'updateWeightsAndBiases', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
		gpu_code = clSetKernelArg(gpu_kernels[3], 2, sizeof(cl_mem), &weights_buffers[i]);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDtrainFromImageListGPU(): Cannot set kernel argument 2 of kernel 'updateWeightsAndBiases', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
		gpu_code = clSetKernelArg(gpu_kernels[3], 3, sizeof(cl_mem), &biases_buffers[i]);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDtrainFromImageListGPU(): Cannot set kernel argument 3 of kernel 'updateWeightsAndBiases', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
		gpu_code = clSetKernelArg(gpu_kernels[3], 4, sizeof(int), &network->layers[i].nb_neurons);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDtrainFromImageListGPU(): Cannot set kernel argument 4 of kernel 'updateWeightsAndBiases', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
		gpu_code = clSetKernelArg(gpu_kernels[3], 5, sizeof(int), &network->layers[i].nb_inputs_per_neuron);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDtrainFromImageListGPU(): Cannot set kernel argument 5 of kernel 'updateWeightsAndBiases', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
		gpu_code = clSetKernelArg(gpu_kernels[3], 6, sizeof(double), &network->learning_rate);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDtrainFromImageListGPU(): Cannot set kernel argument 6 of kernel 'updateWeightsAndBiases', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
		size_t global_work_size[] = {network->layers[i].nb_neurons};
		int backpropagation_index = *event_index - network->nb_layers;
		gpu_code = clEnqueueNDRangeKernel(gpu_oc.command_queue, gpu_kernels[3], 1, NULL, global_work_size, NULL, 1, &events[backpropagation_index], &events[*event_index]);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDtrainFromImageListGPU(): Cannot enqueue kernel 'updateWeightsAndBiases', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
	}

	// Return
	return 0;
}

/**
 * @brief Train the neural network using an image list and the excepted output image
 * For a neural network using double as type and using GPU
 * 
 * @param network			Pointer to the neural network
 * @param img_list			Image list to use for the training
 * @param excepted_output	Excepted output image (must be the same size as the output layer)
 * @param read_all_buffers	Integer, if 1, read all the buffers, if 0, don't read them
 * 
 * @return void
 */
int NeuralNetworkDtrainFromImageListGPU(NeuralNetworkD *network, img_list_t img_list, image_t excepted_output, int read_all_buffers) {

	// Setup the private variables if not already done
	setupNeuralNetworkGpuOpenCL(network);

	// Copy the excepted output image into the output layer of the neural network
	cl_event excepted_output_buffer_event;
	double *excepted_output_array = image_to_double_array(excepted_output, network->output_layer->nb_neurons, 0);
	gpu_code = clEnqueueWriteBuffer(gpu_oc.command_queue, activation_values_buffers[network->nb_layers], CL_FALSE, 0, network->output_layer->nb_neurons * sizeof(double), excepted_output_array, 0, NULL, &excepted_output_buffer_event);
	ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDtrainFromImageListGPU(): Cannot write buffer 'activation_values_buffers[%d]', reason: %d / %s\n", network->nb_layers, gpu_code, getOpenCLErrorString(gpu_code));

	// Allocate memory for cl_event objects, so we need for each image of the image list:
	// - x(1) cl_event for the input layer (write image ratio + image data)
	// - x(nb_layers - 1) cl_event for the feed forward (1 per layer)
	// - x(nb_layers - 1) cl_event for the backpropagation (1 per layer)
	// - x(nb_layers - 1) cl_event for the update weights and biases (1 per layer)
	int total_events = img_list.size * (1 + (network->nb_layers - 1) * 3);
	cl_event *events = malloc(total_events * sizeof(cl_event));

	// For each image of the image list,
	int image_index = 0, event_index = 0;
	img_list_elt_t *current_elt = img_list.head;
	double **images_datas = malloc(img_list.size * sizeof(double*));
	while (current_elt != NULL) {

		// Print benchmark
		#if GPU_TRAINING_BENCHMARK_LEVEL > 0
		char benchmark_buffer[1024];
		char benchmark_name[512];
		sprintf(benchmark_name, "NeuralNetworkDtrainFromImageListGPU(): Image %d / %d", image_index + 1, img_list.size);
		ST_BENCHMARK_SOLO_COUNT(benchmark_buffer, {

		// Call function to train the neural network with the current image
		gpu_code = NeuralNetworkDtrainFromImageListPVImgGPU(network, events, &event_index, current_elt->image, image_index, images_datas, excepted_output_buffer_event);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDtrainFromImageListGPU(): Cannot train the neural network with the current image, reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));

		// Print benchmark
		}, benchmark_name, 1);
		PRINTER(benchmark_buffer);
		#else
			gpu_code = NeuralNetworkDtrainFromImageListPVImgGPU(network, events, &event_index, current_elt->image, image_index, images_datas, excepted_output_buffer_event);
			ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDtrainFromImageListGPU(): Cannot train the neural network with the current image, reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
		#endif

		// Next image
		current_elt = current_elt->next;
		image_index++;
	}

	// Wait for everything to finish
	gpu_code = clFinish(gpu_oc.command_queue);
	ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDtrainFromImageListGPU(): Cannot finish command queue, reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));

	// Read all the buffers if needed
	if (read_all_buffers) {
		gpu_code = NeuralNetworkDReadAllBuffersGPU(network);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDtrainFromImageListGPU(): Cannot read all buffers, reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
	}

	// Release the events
	gpu_code = clReleaseEvent(excepted_output_buffer_event);
	ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDtrainFromImageListGPU(): Cannot release event 'excepted_output_buffer_event', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
	for (int i = 0; i <= event_index; i++) {
		gpu_code = clReleaseEvent(events[i]);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDtrainFromImageListGPU(): Cannot release event 'events[%d]', reason: %d / %s\n", i, gpu_code, getOpenCLErrorString(gpu_code));
	}
	free(events);

	// Free the images datas
	free(excepted_output_array);
	for (int i = 0; i < img_list.size; i++)
		free(images_datas[i]);
	free(images_datas);

	// Return
	return 0;
}


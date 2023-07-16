
#include "training_gpu.h"
#include "../gpu/gpu_utils.h"

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
	activation_values_buffers = (cl_mem*)malloc((network->nb_layers + 1) * sizeof(cl_mem));
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
		weights_buffers[i] = clCreateBuffer(gpu_oc.context, CL_MEM_READ_ONLY, network->layers[i].nb_neurons * network->layers[i].nb_inputs_per_neuron * sizeof(double), NULL, &gpu_code);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "setupNeuralNetworkGpuBuffersOpenCL(): Cannot create buffer 'weights_buffers[%d]' (size: %d = %d * %d), reason: %d / %s\n", i, network->layers[i].nb_neurons * network->layers[i].nb_inputs_per_neuron, network->layers[i].nb_neurons, network->layers[i].nb_inputs_per_neuron, gpu_code, getOpenCLErrorString(gpu_code));

		// Create the biases buffer
		biases_buffers[i] = clCreateBuffer(gpu_oc.context, CL_MEM_READ_ONLY, network->layers[i].nb_neurons * sizeof(double), NULL, &gpu_code);
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

	// Allocate the events
	cl_event *read_events_act_val = (cl_event*)malloc(network->nb_layers * sizeof(cl_event));
	cl_event *read_events_others = (cl_event*)malloc((network->nb_layers - 1) * sizeof(cl_event) * 3);

	// Read all the buffers
	for (int i = 0; i < network->nb_layers; i++) {
		gpu_code = clEnqueueReadBuffer(gpu_oc.command_queue, activation_values_buffers[i], CL_FALSE, 0, network->layers[i].nb_neurons * sizeof(double), network->layers[i].activations_values, 0, NULL, &read_events_act_val[i]);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDtrainGPU(): Cannot read buffer 'activation_values_buffers[%d]', reason: %d / %s\n", i, gpu_code, getOpenCLErrorString(gpu_code));
	}
	for (int i = 1; i < network->nb_layers; i++) {
		gpu_code = clEnqueueReadBuffer(gpu_oc.command_queue, weights_buffers[i], CL_FALSE, 0, network->layers[i].nb_neurons * network->layers[i].nb_inputs_per_neuron * sizeof(double), network->layers[i].weights_flat, 0, NULL, &read_events_others[(i - 1) * 3]);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDtrainGPU(): Cannot read buffer 'weights_buffers[%d]', reason: %d / %s\n", i, gpu_code, getOpenCLErrorString(gpu_code));
		gpu_code = clEnqueueReadBuffer(gpu_oc.command_queue, biases_buffers[i], CL_FALSE, 0, network->layers[i].nb_neurons * sizeof(double), network->layers[i].biases, 0, NULL, &read_events_others[(i - 1) * 3 + 1]);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDtrainGPU(): Cannot read buffer 'biases_buffers[%d]', reason: %d / %s\n", i, gpu_code, getOpenCLErrorString(gpu_code));
		gpu_code = clEnqueueReadBuffer(gpu_oc.command_queue, deltas_buffers[i], CL_FALSE, 0, network->layers[i].nb_neurons * sizeof(double), network->layers[i].deltas, 0, NULL, &read_events_others[(i - 1) * 3 + 2]);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDtrainGPU(): Cannot read buffer 'deltas_buffers[%d]', reason: %d / %s\n", i, gpu_code, getOpenCLErrorString(gpu_code));
	}

	// Wait for all the events
	gpu_code = clWaitForEvents(network->nb_layers, read_events_act_val);
	ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDtrainGPU(): Cannot wait for events 'read_events_act_val[0:%d]', reason: %d / %s\n", network->nb_layers - 1, gpu_code, getOpenCLErrorString(gpu_code));
	gpu_code = clWaitForEvents((network->nb_layers - 1) * 3, read_events_others);
	ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDtrainGPU(): Cannot wait for events 'read_events_others[0:%d]', reason: %d / %s\n", (network->nb_layers - 1) * 3 - 1, gpu_code, getOpenCLErrorString(gpu_code));

	// Release the events
	for (int i = 0; i < network->nb_layers; i++) {
		gpu_code = clReleaseEvent(read_events_act_val[i]);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDtrainGPU(): Cannot release event 'read_events_act_val[%d]', reason: %d / %s\n", i, gpu_code, getOpenCLErrorString(gpu_code));
	}
	for (int i = 0; i < (network->nb_layers - 1) * 3; i++) {
		gpu_code = clReleaseEvent(read_events_others[i]);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDtrainGPU(): Cannot release event 'read_events_others[%d]', reason: %d / %s\n", i, gpu_code, getOpenCLErrorString(gpu_code));
	}
	free(read_events_act_val);
	free(read_events_others);

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

		// For each neuron of the current layer,
		int current_layer_size = network->layers[i].nb_neurons;
		int previous_layer_size = network->layers[i - 1].nb_neurons;
		gpu_code = clSetKernelArg(gpu_kernels[0], 0, sizeof(cl_mem), &activation_values_buffers[i - 1]);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDfeedForwardGPU(): Cannot set kernel argument 0 of kernel 'feedForwardActivationValuesSigmoid', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
		gpu_code = clSetKernelArg(gpu_kernels[0], 1, sizeof(cl_mem), &weights_buffers[i]);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDfeedForwardGPU(): Cannot set kernel argument 1 of kernel 'feedForwardActivationValuesSigmoid', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
		gpu_code = clSetKernelArg(gpu_kernels[0], 2, sizeof(cl_mem), &biases_buffers[i]);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDfeedForwardGPU(): Cannot set kernel argument 2 of kernel 'feedForwardActivationValuesSigmoid', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
		gpu_code = clSetKernelArg(gpu_kernels[0], 3, sizeof(cl_mem), &activation_values_buffers[i]);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDfeedForwardGPU(): Cannot set kernel argument 3 of kernel 'feedForwardActivationValuesSigmoid', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
		gpu_code = clSetKernelArg(gpu_kernels[0], 4, sizeof(int), &current_layer_size);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDfeedForwardGPU(): Cannot set kernel argument 4 of kernel 'feedForwardActivationValuesSigmoid', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
		gpu_code = clSetKernelArg(gpu_kernels[0], 5, sizeof(int), &previous_layer_size);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDfeedForwardGPU(): Cannot set kernel argument 5 of kernel 'feedForwardActivationValuesSigmoid', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
		size_t global_work_size[] = {current_layer_size};
		gpu_code = clEnqueueNDRangeKernel(gpu_oc.command_queue, gpu_kernels[0], 1, NULL, global_work_size, NULL, 0, NULL, NULL);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDfeedForwardGPU(): Cannot enqueue kernel 'feedForwardActivationValuesSigmoid', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
	}

	// Read the activation_values buffer of the output layer if needed
	if (read_buffer) {
		gpu_code = clFinish(gpu_oc.command_queue);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDfeedForwardGPU(): Cannot finish command queue, reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
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
	size_t global_work_size[] = {output_layer_size};
	gpu_code = clEnqueueNDRangeKernel(gpu_oc.command_queue, gpu_kernels[1], 1, NULL, global_work_size, NULL, 0, NULL, NULL);
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
	cl_event excepted_output_buffer_event;
	cl_event *kernel_events = malloc(network->nb_layers * sizeof(cl_event));
	cl_event *backpropagation_events = malloc(network->nb_layers * sizeof(cl_event));
	cl_event wait_list[2];

	// Setup the private variables if not already done
	gpu_code = setupNeuralNetworkGpuOpenCL(network);
	ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDfeedForwardGPU(): Cannot setup GPU, reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));

	// Create a buffer for the excepted output array for the backpropagation later
	gpu_code = clEnqueueWriteBuffer(gpu_oc.command_queue, activation_values_buffers[network->nb_layers], CL_FALSE, 0, network->output_layer->nb_neurons * sizeof(double), excepted_output, 0, NULL, &excepted_output_buffer_event);
	ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDtrainGPU(): Cannot write buffer 'activation_values_buffers[%d]', reason: %d / %s\n", network->nb_layers, gpu_code, getOpenCLErrorString(gpu_code));

	///// Feed forward part /////
	// Set the input layer (copy the input array into the input layer of the neural network)
	gpu_code = clEnqueueWriteBuffer(gpu_oc.command_queue, activation_values_buffers[0], CL_FALSE, 0, network->input_layer->nb_neurons * sizeof(double), input, 0, NULL, &kernel_events[0]);
	ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDfeedForwardGPU(): Cannot write buffer 'activation_values_buffers[0]', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));

	// For each layer of the neural network (except the input layer),
	for (int i = 1; i < network->nb_layers; i++) {

		// For each neuron of the current layer, calculate the activation_value of the neuron
		gpu_code = clSetKernelArg(gpu_kernels[0], 0, sizeof(cl_mem), &activation_values_buffers[i - 1]);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDfeedForwardGPU(): Cannot set kernel argument 0 of kernel 'feedForwardActivationValuesSigmoid', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
		gpu_code = clSetKernelArg(gpu_kernels[0], 1, sizeof(cl_mem), &weights_buffers[i]);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDfeedForwardGPU(): Cannot set kernel argument 1 of kernel 'feedForwardActivationValuesSigmoid', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
		gpu_code = clSetKernelArg(gpu_kernels[0], 2, sizeof(cl_mem), &biases_buffers[i]);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDfeedForwardGPU(): Cannot set kernel argument 2 of kernel 'feedForwardActivationValuesSigmoid', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
		gpu_code = clSetKernelArg(gpu_kernels[0], 3, sizeof(cl_mem), &activation_values_buffers[i]);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDfeedForwardGPU(): Cannot set kernel argument 3 of kernel 'feedForwardActivationValuesSigmoid', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
		gpu_code = clSetKernelArg(gpu_kernels[0], 4, sizeof(int), &network->layers[i].nb_neurons);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDfeedForwardGPU(): Cannot set kernel argument 4 of kernel 'feedForwardActivationValuesSigmoid', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
		gpu_code = clSetKernelArg(gpu_kernels[0], 5, sizeof(int), &network->layers[i].nb_inputs_per_neuron);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDfeedForwardGPU(): Cannot set kernel argument 5 of kernel 'feedForwardActivationValuesSigmoid', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
		size_t global_work_size[] = {network->layers[i].nb_neurons};
		gpu_code = clEnqueueNDRangeKernel(gpu_oc.command_queue, gpu_kernels[0], 1, NULL, global_work_size, NULL, 1, &kernel_events[i - 1], &kernel_events[i]);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDfeedForwardGPU(): Cannot enqueue kernel 'feedForwardActivationValuesSigmoid', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
	}

	///// Backpropagation part /////
	// Prepare wait list for the last kernel & output buffer
	wait_list[0] = kernel_events[network->nb_layers - 1];
	wait_list[1] = excepted_output_buffer_event;

	// For each neuron of the output layer, calculate the delta of the neuron (excepted_output - activation_value) * derivative
	gpu_code = clSetKernelArg(gpu_kernels[1], 0, sizeof(cl_mem), &activation_values_buffers[network->nb_layers - 1]);
	ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDbackpropagationGPU(): Cannot set kernel argument 0 of kernel 'backpropagationOutputLayerDeltas', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
	gpu_code = clSetKernelArg(gpu_kernels[1], 1, sizeof(cl_mem), &activation_values_buffers[network->nb_layers - 1]);
	ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDbackpropagationGPU(): Cannot set kernel argument 1 of kernel 'backpropagationOutputLayerDeltas', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
	gpu_code = clSetKernelArg(gpu_kernels[1], 2, sizeof(cl_mem), &deltas_buffers[network->nb_layers - 1]);
	ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDbackpropagationGPU(): Cannot set kernel argument 2 of kernel 'backpropagationOutputLayerDeltas', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
	gpu_code = clSetKernelArg(gpu_kernels[1], 3, sizeof(int), &network->output_layer->nb_neurons);
	ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDbackpropagationGPU(): Cannot set kernel argument 3 of kernel 'backpropagationOutputLayerDeltas', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
	size_t global_work_size[] = {network->output_layer->nb_neurons};
	gpu_code = clEnqueueNDRangeKernel(gpu_oc.command_queue, gpu_kernels[1], 1, NULL, global_work_size, NULL, 2, wait_list, &backpropagation_events[network->nb_layers - 1]);
	ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDbackpropagationGPU(): Cannot enqueue kernel 'backpropagationOutputLayerDeltas', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));

	// For each layer of the neural network (except the output layer and the input layer) (order last to first),
	for (int i = network->nb_layers - 2; i > 0; i--) {

		// For each neuron of the current layer,
		gpu_code = clSetKernelArg(gpu_kernels[2], 0, sizeof(cl_mem), &weights_buffers[i + 1]);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDbackpropagationGPU(): Cannot set kernel argument 0 of kernel 'backpropagationHiddenLayersDeltas', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
		gpu_code = clSetKernelArg(gpu_kernels[2], 1, sizeof(cl_mem), &deltas_buffers[i + 1]);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDbackpropagationGPU(): Cannot set kernel argument 1 of kernel 'backpropagationHiddenLayersDeltas', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
		gpu_code = clSetKernelArg(gpu_kernels[2], 2, sizeof(cl_mem), &activation_values_buffers[i]);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDbackpropagationGPU(): Cannot set kernel argument 2 of kernel 'backpropagationHiddenLayersDeltas', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
		gpu_code = clSetKernelArg(gpu_kernels[2], 3, sizeof(cl_mem), &deltas_buffers[i]);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDbackpropagationGPU(): Cannot set kernel argument 3 of kernel 'backpropagationHiddenLayersDeltas', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
		gpu_code = clSetKernelArg(gpu_kernels[2], 4, sizeof(int), &network->layers[i].nb_neurons);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDbackpropagationGPU(): Cannot set kernel argument 4 of kernel 'backpropagationHiddenLayersDeltas', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
		gpu_code = clSetKernelArg(gpu_kernels[2], 5, sizeof(int), &network->layers[i + 1].nb_neurons);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDbackpropagationGPU(): Cannot set kernel argument 5 of kernel 'backpropagationHiddenLayersDeltas', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
		size_t global_work_size[] = {network->layers[i].nb_neurons};
		gpu_code = clEnqueueNDRangeKernel(gpu_oc.command_queue, gpu_kernels[2], 1, NULL, global_work_size, NULL, 1, &backpropagation_events[i + 1], &backpropagation_events[i]);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDbackpropagationGPU(): Cannot enqueue kernel 'backpropagationHiddenLayersDeltas', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
	}

	///// Update weights and biases part /////
	// For each layer of the neural network (except the input layer) (order last to first),
	for (int i = network->nb_layers - 1; i > 0; i--) {

		// For each neuron of the current layer,
		gpu_code = clSetKernelArg(gpu_kernels[3], 0, sizeof(cl_mem), &activation_values_buffers[i - 1]);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDupdateWeightsAndBiasesGPU(): Cannot set kernel argument 0 of kernel 'updateWeightsAndBiases', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
		gpu_code = clSetKernelArg(gpu_kernels[3], 1, sizeof(cl_mem), &deltas_buffers[i]);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDupdateWeightsAndBiasesGPU(): Cannot set kernel argument 1 of kernel 'updateWeightsAndBiases', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
		gpu_code = clSetKernelArg(gpu_kernels[3], 2, sizeof(cl_mem), &weights_buffers[i]);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDupdateWeightsAndBiasesGPU(): Cannot set kernel argument 2 of kernel 'updateWeightsAndBiases', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
		gpu_code = clSetKernelArg(gpu_kernels[3], 3, sizeof(cl_mem), &biases_buffers[i]);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDupdateWeightsAndBiasesGPU(): Cannot set kernel argument 3 of kernel 'updateWeightsAndBiases', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
		gpu_code = clSetKernelArg(gpu_kernels[3], 4, sizeof(int), &network->layers[i].nb_neurons);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDupdateWeightsAndBiasesGPU(): Cannot set kernel argument 4 of kernel 'updateWeightsAndBiases', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
		gpu_code = clSetKernelArg(gpu_kernels[3], 5, sizeof(int), &network->layers[i].nb_inputs_per_neuron);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDupdateWeightsAndBiasesGPU(): Cannot set kernel argument 5 of kernel 'updateWeightsAndBiases', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
		gpu_code = clSetKernelArg(gpu_kernels[3], 6, sizeof(double), &network->learning_rate);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDupdateWeightsAndBiasesGPU(): Cannot set kernel argument 6 of kernel 'updateWeightsAndBiases', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
		size_t global_work_size[] = {network->layers[i].nb_neurons};
		gpu_code = clEnqueueNDRangeKernel(gpu_oc.command_queue, gpu_kernels[3], 1, NULL, global_work_size, NULL, 1, &backpropagation_events[i], NULL);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDupdateWeightsAndBiasesGPU(): Cannot enqueue kernel 'updateWeightsAndBiases', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
	}

	// Wait for everything to finish
	gpu_code = clFinish(gpu_oc.command_queue);
	ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDtrainGPU(): Cannot finish command queue, reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));

	// Read all the buffers if needed
	if (read_all_buffers) {
		gpu_code = NeuralNetworkDReadAllBuffersGPU(network);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDtrainGPU(): Cannot read all buffers, reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
	}

	// Release the events
	gpu_code = clReleaseEvent(excepted_output_buffer_event);
	ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDtrainGPU(): Cannot release event 'excepted_output_buffer_event', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
	for (int i = 0; i < network->nb_layers; i++) {
		gpu_code = clReleaseEvent(kernel_events[i]);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDtrainGPU(): Cannot release event 'kernel_events[%d]', reason: %d / %s\n", i, gpu_code, getOpenCLErrorString(gpu_code));
		if (i == 0)
			continue;
		gpu_code = clReleaseEvent(backpropagation_events[i]);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDtrainGPU(): Cannot release event 'backpropagation_events[%d]', reason: %d / %s\n", i, gpu_code, getOpenCLErrorString(gpu_code));
	}
	free(kernel_events);
	free(backpropagation_events);

	// Return
	return 0;
}

// Struct for the events
typedef struct events_t {
	cl_event *kernel_events;
	cl_event *backpropagation_events;
	cl_event wait_list[2];
	cl_event done;
} events_t;

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

	// Setup the events
	cl_event excepted_output_buffer_event;
	cl_event *img_list_events = malloc(img_list.size * sizeof(cl_event));
	events_t *events = malloc(sizeof(events_t));
	for (int i = 0; i < img_list.size; i++) {
		events[i].kernel_events = malloc(network->nb_layers * sizeof(cl_event));
		events[i].backpropagation_events = malloc(network->nb_layers * sizeof(cl_event));
	}

	// Create a buffer for the excepted output image for the backpropagation later
	double *excepted_output_flat = image_to_double_array(excepted_output);
	clEnqueueWriteBuffer(gpu_oc.command_queue, activation_values_buffers[network->nb_layers], CL_FALSE, 0, network->output_layer->nb_neurons * sizeof(double), excepted_output_flat, 0, NULL, &excepted_output_buffer_event);

	// For each image of the image list,
	double **img_list_flat = malloc(img_list.size * sizeof(double*));
	img_list_elt_t *current_input = img_list.head;
	int index = 0;
	while (current_input != NULL) {
		DEBUG_PRINT("NeuralNetworkDtrainFromImageListGPU(): Image %d\n", index);

		// Convert the image to a flat array of double and write it to the GPU
		int image_size = current_input->image.width * current_input->image.height * current_input->image.channels;
		img_list_flat[index] = image_to_double_array(current_input->image);
		if (index == 0)
			clEnqueueWriteBuffer(gpu_oc.command_queue, activation_values_buffers[0], CL_FALSE, 0, image_size * sizeof(double), img_list_flat[index], 0, NULL, &img_list_events[index]);
		else
			clEnqueueWriteBuffer(gpu_oc.command_queue, activation_values_buffers[0], CL_FALSE, 0, image_size * sizeof(double), img_list_flat[index], 1, &events[index].done, &img_list_events[index]);
DEBUG_PRINT("NeuralNetworkDtrainFromImageListGPU(): Image %d written\n", index);
		///// Feed forward part /////
		// For each layer of the neural network (except the input layer),
		for (int i = 1; i < network->nb_layers; i++) {

			// For each neuron of the current layer, calculate the activation_value of the neuron
			clSetKernelArg(gpu_kernels[0], 0, sizeof(cl_mem), &activation_values_buffers[i - 1]);
			clSetKernelArg(gpu_kernels[0], 1, sizeof(cl_mem), &weights_buffers[i]);
			clSetKernelArg(gpu_kernels[0], 2, sizeof(cl_mem), &biases_buffers[i]);
			clSetKernelArg(gpu_kernels[0], 3, sizeof(cl_mem), &activation_values_buffers[i]);
			clSetKernelArg(gpu_kernels[0], 4, sizeof(int), &network->layers[i].nb_neurons);
			clSetKernelArg(gpu_kernels[0], 5, sizeof(int), &network->layers[i].nb_inputs_per_neuron);
			size_t global_work_size[] = {network->layers[i].nb_neurons};
			clEnqueueNDRangeKernel(gpu_oc.command_queue, gpu_kernels[0], 1, NULL, global_work_size, NULL, 1, &events[index].kernel_events[i - 1], &events[index].kernel_events[i]);
		}
DEBUG_PRINT("NeuralNetworkDtrainFromImageListGPU(): Feed forward done\n");
		///// Backpropagation part /////
		// Prepare wait list for the last kernel & output buffer
		// wait_list[0] = kernel_events[network->nb_layers - 1];
		// wait_list[1] = excepted_output_buffer_event;
		events[index].wait_list[0] = events[index].kernel_events[network->nb_layers - 1];
		events[index].wait_list[1] = excepted_output_buffer_event;

		// For each neuron of the output layer, calculate the delta of the neuron (excepted_output - activation_value) * derivative
		clSetKernelArg(gpu_kernels[1], 0, sizeof(cl_mem), &activation_values_buffers[network->nb_layers - 1]);
		clSetKernelArg(gpu_kernels[1], 1, sizeof(cl_mem), &activation_values_buffers[network->nb_layers - 1]);
		clSetKernelArg(gpu_kernels[1], 2, sizeof(cl_mem), &deltas_buffers[network->nb_layers - 1]);
		clSetKernelArg(gpu_kernels[1], 3, sizeof(int), &network->output_layer->nb_neurons);
		size_t global_work_size[] = {network->output_layer->nb_neurons};
		clEnqueueNDRangeKernel(gpu_oc.command_queue, gpu_kernels[1], 1, NULL, global_work_size, NULL, 2, events[index].wait_list, &events[index].backpropagation_events[network->nb_layers - 1]);
DEBUG_PRINT("NeuralNetworkDtrainFromImageListGPU(): Backpropagation output layer done\n");
		// For each layer of the neural network (except the output layer and the input layer) (order last to first),
		for (int i = network->nb_layers - 2; i > 0; i--) {

			// For each neuron of the current layer,
			clSetKernelArg(gpu_kernels[2], 0, sizeof(cl_mem), &weights_buffers[i + 1]);
			clSetKernelArg(gpu_kernels[2], 1, sizeof(cl_mem), &deltas_buffers[i + 1]);
			clSetKernelArg(gpu_kernels[2], 2, sizeof(cl_mem), &activation_values_buffers[i]);
			clSetKernelArg(gpu_kernels[2], 3, sizeof(cl_mem), &deltas_buffers[i]);
			clSetKernelArg(gpu_kernels[2], 4, sizeof(int), &network->layers[i].nb_neurons);
			clSetKernelArg(gpu_kernels[2], 5, sizeof(int), &network->layers[i + 1].nb_neurons);
			size_t global_work_size[] = {network->layers[i].nb_neurons};
			clEnqueueNDRangeKernel(gpu_oc.command_queue, gpu_kernels[2], 1, NULL, global_work_size, NULL, 1, &events[index].backpropagation_events[i + 1], &events[index].backpropagation_events[i]);
		}
DEBUG_PRINT("NeuralNetworkDtrainFromImageListGPU(): Backpropagation hidden layers done\n");
		///// Update weights and biases part /////
		// For each layer of the neural network (except the input layer) (order last to first),
		for (int i = network->nb_layers - 1; i > 0; i--) {

			// For each neuron of the current layer,
			clSetKernelArg(gpu_kernels[3], 0, sizeof(cl_mem), &activation_values_buffers[i - 1]);
			clSetKernelArg(gpu_kernels[3], 1, sizeof(cl_mem), &deltas_buffers[i]);
			clSetKernelArg(gpu_kernels[3], 2, sizeof(cl_mem), &weights_buffers[i]);
			clSetKernelArg(gpu_kernels[3], 3, sizeof(cl_mem), &biases_buffers[i]);
			clSetKernelArg(gpu_kernels[3], 4, sizeof(int), &network->layers[i].nb_neurons);
			clSetKernelArg(gpu_kernels[3], 5, sizeof(int), &network->layers[i].nb_inputs_per_neuron);
			clSetKernelArg(gpu_kernels[3], 6, sizeof(double), &network->learning_rate);
			size_t global_work_size[] = {network->layers[i].nb_neurons};
			if (i == 1)
				clEnqueueNDRangeKernel(gpu_oc.command_queue, gpu_kernels[3], 1, NULL, global_work_size, NULL, 1, &events[index].backpropagation_events[i], &events[index].done);
			else
				clEnqueueNDRangeKernel(gpu_oc.command_queue, gpu_kernels[3], 1, NULL, global_work_size, NULL, 1, &events[index].backpropagation_events[i], NULL);
		}

		// Increment the index and the current_input
		index++;
		current_input = current_input->next;
		DEBUG_PRINT("NeuralNetworkDtrainFromImageListGPU(): %d / %d\n", index, img_list.size);
	}

	// Wait for everything to finish
	clFinish(gpu_oc.command_queue);

	// Read all the buffers if needed
	if (read_all_buffers) {
		NeuralNetworkDReadAllBuffersGPU(network);
	}

	// Release the events
	clReleaseEvent(excepted_output_buffer_event);
	for (int i = 0; i < img_list.size; i++) {
		clReleaseEvent(img_list_events[i]);
		for (int j = 0; j < network->nb_layers; j++) {
			clReleaseEvent(events[i].kernel_events[j]);
			if (j == 0)
				continue;
			clReleaseEvent(events[i].backpropagation_events[j]);
		}
		free(events[i].kernel_events);
		free(events[i].backpropagation_events);
	}
	free(img_list_events);
	free(events);

	// Free the flat arrays
	free(excepted_output_flat);
	for (int i = 0; i < img_list.size; i++)
		free(img_list_flat[i]);
	free(img_list_flat);
DEBUG_PRINT("NeuralNetworkDtrainFromImageListGPU(): End\n");
	// Return
	return 0;
}

